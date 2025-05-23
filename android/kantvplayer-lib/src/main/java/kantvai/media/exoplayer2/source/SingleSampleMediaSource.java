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
package kantvai.media.exoplayer2.source;

import static kantvai.media.exoplayer2.util.Assertions.checkNotNull;

import android.net.Uri;
import androidx.annotation.Nullable;
import kantvai.media.exoplayer2.Format;
import kantvai.media.exoplayer2.MediaItem;
import kantvai.media.exoplayer2.Timeline;
import kantvai.media.exoplayer2.upstream.Allocator;
import kantvai.media.exoplayer2.upstream.DataSource;
import kantvai.media.exoplayer2.upstream.DataSpec;
import kantvai.media.exoplayer2.upstream.DefaultLoadErrorHandlingPolicy;
import kantvai.media.exoplayer2.upstream.LoadErrorHandlingPolicy;
import kantvai.media.exoplayer2.upstream.TransferListener;
import java.util.Collections;

/**
 * Loads data at a given {@link Uri} as a single sample belonging to a single {@link MediaPeriod}.
 */
public final class SingleSampleMediaSource extends BaseMediaSource {

  /** Factory for {@link SingleSampleMediaSource}. */
  public static final class Factory {

    private final DataSource.Factory dataSourceFactory;

    private LoadErrorHandlingPolicy loadErrorHandlingPolicy;
    private boolean treatLoadErrorsAsEndOfStream;
    @Nullable private Object tag;
    @Nullable private String trackId;

    /**
     * Creates a factory for {@link SingleSampleMediaSource}s.
     *
     * @param dataSourceFactory The factory from which the {@link DataSource} to read the media will
     *     be obtained.
     */
    public Factory(DataSource.Factory dataSourceFactory) {
      this.dataSourceFactory = checkNotNull(dataSourceFactory);
      loadErrorHandlingPolicy = new DefaultLoadErrorHandlingPolicy();
      treatLoadErrorsAsEndOfStream = true;
    }

    /**
     * Sets a tag for the media source which will be published in the {@link Timeline} of the source
     * as {@link kantvai.media.exoplayer2.MediaItem.PlaybackProperties#tag
     * Window#mediaItem.playbackProperties.tag}.
     *
     * @param tag A tag for the media source.
     * @return This factory, for convenience.
     */
    public Factory setTag(@Nullable Object tag) {
      this.tag = tag;
      return this;
    }

    /**
     * Sets an optional track id to be used.
     *
     * @param trackId An optional track id.
     * @return This factory, for convenience.
     */
    public Factory setTrackId(@Nullable String trackId) {
      this.trackId = trackId;
      return this;
    }

    /**
     * Sets the {@link LoadErrorHandlingPolicy}. The default value is created by calling {@link
     * DefaultLoadErrorHandlingPolicy#DefaultLoadErrorHandlingPolicy()}.
     *
     * @param loadErrorHandlingPolicy A {@link LoadErrorHandlingPolicy}.
     * @return This factory, for convenience.
     */
    public Factory setLoadErrorHandlingPolicy(
        @Nullable LoadErrorHandlingPolicy loadErrorHandlingPolicy) {
      this.loadErrorHandlingPolicy =
          loadErrorHandlingPolicy != null
              ? loadErrorHandlingPolicy
              : new DefaultLoadErrorHandlingPolicy();
      return this;
    }

    /**
     * Sets whether load errors will be treated as end-of-stream signal (load errors will not be
     * propagated). The default value is true.
     *
     * @param treatLoadErrorsAsEndOfStream If true, load errors will not be propagated by sample
     *     streams, treating them as ended instead. If false, load errors will be propagated
     *     normally by {@link SampleStream#maybeThrowError()}.
     * @return This factory, for convenience.
     */
    public Factory setTreatLoadErrorsAsEndOfStream(boolean treatLoadErrorsAsEndOfStream) {
      this.treatLoadErrorsAsEndOfStream = treatLoadErrorsAsEndOfStream;
      return this;
    }

    /**
     * Returns a new {@link SingleSampleMediaSource} using the current parameters.
     *
     * @param subtitle The {@link MediaItem.Subtitle}.
     * @param durationUs The duration of the media stream in microseconds.
     * @return The new {@link SingleSampleMediaSource}.
     */
    public SingleSampleMediaSource createMediaSource(MediaItem.Subtitle subtitle, long durationUs) {
      return new SingleSampleMediaSource(
          trackId,
          subtitle,
          dataSourceFactory,
          durationUs,
          loadErrorHandlingPolicy,
          treatLoadErrorsAsEndOfStream,
          tag);
    }

    /** @deprecated Use {@link #createMediaSource(MediaItem.Subtitle, long)} instead. */
    @Deprecated
    public SingleSampleMediaSource createMediaSource(Uri uri, Format format, long durationUs) {
      return new SingleSampleMediaSource(
          format.id == null ? trackId : format.id,
          new MediaItem.Subtitle(
              uri, checkNotNull(format.sampleMimeType), format.language, format.selectionFlags),
          dataSourceFactory,
          durationUs,
          loadErrorHandlingPolicy,
          treatLoadErrorsAsEndOfStream,
          tag);
    }
  }

  private final DataSpec dataSpec;
  private final DataSource.Factory dataSourceFactory;
  private final Format format;
  private final long durationUs;
  private final LoadErrorHandlingPolicy loadErrorHandlingPolicy;
  private final boolean treatLoadErrorsAsEndOfStream;
  private final Timeline timeline;
  private final MediaItem mediaItem;

  @Nullable private TransferListener transferListener;

  private SingleSampleMediaSource(
      @Nullable String trackId,
      MediaItem.Subtitle subtitle,
      DataSource.Factory dataSourceFactory,
      long durationUs,
      LoadErrorHandlingPolicy loadErrorHandlingPolicy,
      boolean treatLoadErrorsAsEndOfStream,
      @Nullable Object tag) {
    this.dataSourceFactory = dataSourceFactory;
    this.durationUs = durationUs;
    this.loadErrorHandlingPolicy = loadErrorHandlingPolicy;
    this.treatLoadErrorsAsEndOfStream = treatLoadErrorsAsEndOfStream;
    mediaItem =
        new MediaItem.Builder()
            .setUri(Uri.EMPTY)
            .setMediaId(subtitle.uri.toString())
            .setSubtitles(Collections.singletonList(subtitle))
            .setTag(tag)
            .build();
    format =
        new Format.Builder()
            .setId(trackId)
            .setSampleMimeType(subtitle.mimeType)
            .setLanguage(subtitle.language)
            .setSelectionFlags(subtitle.selectionFlags)
            .setRoleFlags(subtitle.roleFlags)
            .setLabel(subtitle.label)
            .build();
    dataSpec =
        new DataSpec.Builder().setUri(subtitle.uri).setFlags(DataSpec.FLAG_ALLOW_GZIP).build();
    timeline =
        new SinglePeriodTimeline(
            durationUs,
            /* isSeekable= */ true,
            /* isDynamic= */ false,
            /* useLiveConfiguration= */ false,
            /* manifest= */ null,
            mediaItem);
  }

  // MediaSource implementation.

  @Override
  public MediaItem getMediaItem() {
    return mediaItem;
  }

  @Override
  protected void prepareSourceInternal(@Nullable TransferListener mediaTransferListener) {
    transferListener = mediaTransferListener;
    refreshSourceInfo(timeline);
  }

  @Override
  public void maybeThrowSourceInfoRefreshError() {
    // Do nothing.
  }

  @Override
  public MediaPeriod createPeriod(MediaPeriodId id, Allocator allocator, long startPositionUs) {
    return new SingleSampleMediaPeriod(
        dataSpec,
        dataSourceFactory,
        transferListener,
        format,
        durationUs,
        loadErrorHandlingPolicy,
        createEventDispatcher(id),
        treatLoadErrorsAsEndOfStream);
  }

  @Override
  public void releasePeriod(MediaPeriod mediaPeriod) {
    ((SingleSampleMediaPeriod) mediaPeriod).release();
  }

  @Override
  protected void releaseSourceInternal() {
    // Do nothing.
  }
}
