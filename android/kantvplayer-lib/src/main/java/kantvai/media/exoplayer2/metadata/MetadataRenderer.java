/*
 * Copyright (C) 2016 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package kantvai.media.exoplayer2.metadata;

import static kantvai.media.exoplayer2.util.Util.castNonNull;

import android.os.Handler;
import android.os.Handler.Callback;
import android.os.Looper;
import android.os.Message;
import androidx.annotation.Nullable;
import kantvai.media.exoplayer2.BaseRenderer;
import kantvai.media.exoplayer2.C;
import kantvai.media.exoplayer2.Format;
import kantvai.media.exoplayer2.FormatHolder;
import kantvai.media.exoplayer2.RendererCapabilities;
import kantvai.media.exoplayer2.source.SampleStream.ReadDataResult;
import kantvai.media.exoplayer2.util.Assertions;
import kantvai.media.exoplayer2.util.Util;
import java.util.ArrayList;
import java.util.List;

/** A renderer for metadata. */
public final class MetadataRenderer extends BaseRenderer implements Callback {

  private static final String TAG = "MetadataRenderer";
  private static final int MSG_INVOKE_RENDERER = 0;

  private final MetadataDecoderFactory decoderFactory;
  private final MetadataOutput output;
  @Nullable private final Handler outputHandler;
  private final MetadataInputBuffer buffer;

  @Nullable private MetadataDecoder decoder;
  private boolean inputStreamEnded;
  private boolean outputStreamEnded;
  private long subsampleOffsetUs;
  private long pendingMetadataTimestampUs;
  @Nullable private Metadata pendingMetadata;

  /**
   * @param output The output.
   * @param outputLooper The looper associated with the thread on which the output should be called.
   *     If the output makes use of standard Android UI components, then this should normally be the
   *     looper associated with the application's main thread, which can be obtained using {@link
   *     android.app.Activity#getMainLooper()}. Null may be passed if the output should be called
   *     directly on the player's internal rendering thread.
   */
  public MetadataRenderer(MetadataOutput output, @Nullable Looper outputLooper) {
    this(output, outputLooper, MetadataDecoderFactory.DEFAULT);
  }

  /**
   * @param output The output.
   * @param outputLooper The looper associated with the thread on which the output should be called.
   *     If the output makes use of standard Android UI components, then this should normally be the
   *     looper associated with the application's main thread, which can be obtained using {@link
   *     android.app.Activity#getMainLooper()}. Null may be passed if the output should be called
   *     directly on the player's internal rendering thread.
   * @param decoderFactory A factory from which to obtain {@link MetadataDecoder} instances.
   */
  public MetadataRenderer(
      MetadataOutput output, @Nullable Looper outputLooper, MetadataDecoderFactory decoderFactory) {
    super(C.TRACK_TYPE_METADATA);
    this.output = Assertions.checkNotNull(output);
    this.outputHandler =
        outputLooper == null ? null : Util.createHandler(outputLooper, /* callback= */ this);
    this.decoderFactory = Assertions.checkNotNull(decoderFactory);
    buffer = new MetadataInputBuffer();
    pendingMetadataTimestampUs = C.TIME_UNSET;
  }

  @Override
  public String getName() {
    return TAG;
  }

  @Override
  @Capabilities
  public int supportsFormat(Format format) {
    if (decoderFactory.supportsFormat(format)) {
      return RendererCapabilities.create(
          format.exoMediaCryptoType == null ? C.FORMAT_HANDLED : C.FORMAT_UNSUPPORTED_DRM);
    } else {
      return RendererCapabilities.create(C.FORMAT_UNSUPPORTED_TYPE);
    }
  }

  @Override
  protected void onStreamChanged(Format[] formats, long startPositionUs, long offsetUs) {
    decoder = decoderFactory.createDecoder(formats[0]);
  }

  @Override
  protected void onPositionReset(long positionUs, boolean joining) {
    pendingMetadata = null;
    pendingMetadataTimestampUs = C.TIME_UNSET;
    inputStreamEnded = false;
    outputStreamEnded = false;
  }

  @Override
  public void render(long positionUs, long elapsedRealtimeUs) {
    boolean working = true;
    while (working) {
      readMetadata();
      working = outputMetadata(positionUs);
    }
  }

  /**
   * Iterates through {@code metadata.entries} and checks each one to see if contains wrapped
   * metadata. If it does, then we recursively decode the wrapped metadata. If it doesn't (recursion
   * base-case), we add the {@link Metadata.Entry} to {@code decodedEntries} (output parameter).
   */
  private void decodeWrappedMetadata(Metadata metadata, List<Metadata.Entry> decodedEntries) {
    for (int i = 0; i < metadata.length(); i++) {
      @Nullable Format wrappedMetadataFormat = metadata.get(i).getWrappedMetadataFormat();
      if (wrappedMetadataFormat != null && decoderFactory.supportsFormat(wrappedMetadataFormat)) {
        MetadataDecoder wrappedMetadataDecoder =
            decoderFactory.createDecoder(wrappedMetadataFormat);
        // wrappedMetadataFormat != null so wrappedMetadataBytes must be non-null too.
        byte[] wrappedMetadataBytes =
            Assertions.checkNotNull(metadata.get(i).getWrappedMetadataBytes());
        buffer.clear();
        buffer.ensureSpaceForWrite(wrappedMetadataBytes.length);
        castNonNull(buffer.data).put(wrappedMetadataBytes);
        buffer.flip();
        @Nullable Metadata innerMetadata = wrappedMetadataDecoder.decode(buffer);
        if (innerMetadata != null) {
          // The decoding succeeded, so we'll try another level of unwrapping.
          decodeWrappedMetadata(innerMetadata, decodedEntries);
        }
      } else {
        // Entry doesn't contain any wrapped metadata, so output it directly.
        decodedEntries.add(metadata.get(i));
      }
    }
  }

  @Override
  protected void onDisabled() {
    pendingMetadata = null;
    pendingMetadataTimestampUs = C.TIME_UNSET;
    decoder = null;
  }

  @Override
  public boolean isEnded() {
    return outputStreamEnded;
  }

  @Override
  public boolean isReady() {
    return true;
  }

  @Override
  public boolean handleMessage(Message msg) {
    switch (msg.what) {
      case MSG_INVOKE_RENDERER:
        invokeRendererInternal((Metadata) msg.obj);
        return true;
      default:
        // Should never happen.
        throw new IllegalStateException();
    }
  }

  private void readMetadata() {
    if (!inputStreamEnded && pendingMetadata == null) {
      buffer.clear();
      FormatHolder formatHolder = getFormatHolder();
      @ReadDataResult int result = readSource(formatHolder, buffer, /* readFlags= */ 0);
      if (result == C.RESULT_BUFFER_READ) {
        if (buffer.isEndOfStream()) {
          inputStreamEnded = true;
        } else {
          buffer.subsampleOffsetUs = subsampleOffsetUs;
          buffer.flip();
          @Nullable Metadata metadata = castNonNull(decoder).decode(buffer);
          if (metadata != null) {
            List<Metadata.Entry> entries = new ArrayList<>(metadata.length());
            decodeWrappedMetadata(metadata, entries);
            if (!entries.isEmpty()) {
              Metadata expandedMetadata = new Metadata(entries);
              pendingMetadata = expandedMetadata;
              pendingMetadataTimestampUs = buffer.timeUs;
            }
          }
        }
      } else if (result == C.RESULT_FORMAT_READ) {
        subsampleOffsetUs = Assertions.checkNotNull(formatHolder.format).subsampleOffsetUs;
      }
    }
  }

  private boolean outputMetadata(long positionUs) {
    boolean didOutput = false;
    if (pendingMetadata != null && pendingMetadataTimestampUs <= positionUs) {
      invokeRenderer(pendingMetadata);
      pendingMetadata = null;
      pendingMetadataTimestampUs = C.TIME_UNSET;
      didOutput = true;
    }
    if (inputStreamEnded && pendingMetadata == null) {
      outputStreamEnded = true;
    }
    return didOutput;
  }

  private void invokeRenderer(Metadata metadata) {
    if (outputHandler != null) {
      outputHandler.obtainMessage(MSG_INVOKE_RENDERER, metadata).sendToTarget();
    } else {
      invokeRendererInternal(metadata);
    }
  }

  private void invokeRendererInternal(Metadata metadata) {
    output.onMetadata(metadata);
  }
}
