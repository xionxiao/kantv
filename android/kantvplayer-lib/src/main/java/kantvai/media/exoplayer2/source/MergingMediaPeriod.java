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

import static java.lang.Math.max;

import androidx.annotation.Nullable;
import kantvai.media.exoplayer2.C;
import kantvai.media.exoplayer2.FormatHolder;
import kantvai.media.exoplayer2.SeekParameters;
import kantvai.media.exoplayer2.decoder.DecoderInputBuffer;
import kantvai.media.exoplayer2.offline.StreamKey;
import kantvai.media.exoplayer2.trackselection.ExoTrackSelection;
import kantvai.media.exoplayer2.util.Assertions;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import org.checkerframework.checker.nullness.compatqual.NullableType;
import org.checkerframework.checker.nullness.qual.MonotonicNonNull;

/** Merges multiple {@link MediaPeriod}s. */
/* package */ final class MergingMediaPeriod implements MediaPeriod, MediaPeriod.Callback {

  private final MediaPeriod[] periods;
  private final IdentityHashMap<SampleStream, Integer> streamPeriodIndices;
  private final CompositeSequenceableLoaderFactory compositeSequenceableLoaderFactory;
  private final ArrayList<MediaPeriod> childrenPendingPreparation;

  @Nullable private Callback callback;
  @Nullable private TrackGroupArray trackGroups;
  private MediaPeriod[] enabledPeriods;
  private SequenceableLoader compositeSequenceableLoader;

  public MergingMediaPeriod(
      CompositeSequenceableLoaderFactory compositeSequenceableLoaderFactory,
      long[] periodTimeOffsetsUs,
      MediaPeriod... periods) {
    this.compositeSequenceableLoaderFactory = compositeSequenceableLoaderFactory;
    this.periods = periods;
    childrenPendingPreparation = new ArrayList<>();
    compositeSequenceableLoader =
        compositeSequenceableLoaderFactory.createCompositeSequenceableLoader();
    streamPeriodIndices = new IdentityHashMap<>();
    enabledPeriods = new MediaPeriod[0];
    for (int i = 0; i < periods.length; i++) {
      if (periodTimeOffsetsUs[i] != 0) {
        this.periods[i] = new TimeOffsetMediaPeriod(periods[i], periodTimeOffsetsUs[i]);
      }
    }
  }

  /**
   * Returns the child period passed to {@link
   * #MergingMediaPeriod(CompositeSequenceableLoaderFactory, long[], MediaPeriod...)} at the
   * specified index.
   */
  public MediaPeriod getChildPeriod(int index) {
    return periods[index] instanceof TimeOffsetMediaPeriod
        ? ((TimeOffsetMediaPeriod) periods[index]).mediaPeriod
        : periods[index];
  }

  @Override
  public void prepare(Callback callback, long positionUs) {
    this.callback = callback;
    Collections.addAll(childrenPendingPreparation, periods);
    for (MediaPeriod period : periods) {
      period.prepare(this, positionUs);
    }
  }

  @Override
  public void maybeThrowPrepareError() throws IOException {
    for (MediaPeriod period : periods) {
      period.maybeThrowPrepareError();
    }
  }

  @Override
  public TrackGroupArray getTrackGroups() {
    return Assertions.checkNotNull(trackGroups);
  }

  @Override
  public long selectTracks(
      @NullableType ExoTrackSelection[] selections,
      boolean[] mayRetainStreamFlags,
      @NullableType SampleStream[] streams,
      boolean[] streamResetFlags,
      long positionUs) {
    // Map each selection and stream onto a child period index.
    int[] streamChildIndices = new int[selections.length];
    int[] selectionChildIndices = new int[selections.length];
    for (int i = 0; i < selections.length; i++) {
      Integer streamChildIndex = streams[i] == null ? null : streamPeriodIndices.get(streams[i]);
      streamChildIndices[i] = streamChildIndex == null ? C.INDEX_UNSET : streamChildIndex;
      selectionChildIndices[i] = C.INDEX_UNSET;
      if (selections[i] != null) {
        TrackGroup trackGroup = selections[i].getTrackGroup();
        for (int j = 0; j < periods.length; j++) {
          if (periods[j].getTrackGroups().indexOf(trackGroup) != C.INDEX_UNSET) {
            selectionChildIndices[i] = j;
            break;
          }
        }
      }
    }
    streamPeriodIndices.clear();
    // Select tracks for each child, copying the resulting streams back into a new streams array.
    @NullableType SampleStream[] newStreams = new SampleStream[selections.length];
    @NullableType SampleStream[] childStreams = new SampleStream[selections.length];
    @NullableType ExoTrackSelection[] childSelections = new ExoTrackSelection[selections.length];
    ArrayList<MediaPeriod> enabledPeriodsList = new ArrayList<>(periods.length);
    for (int i = 0; i < periods.length; i++) {
      for (int j = 0; j < selections.length; j++) {
        childStreams[j] = streamChildIndices[j] == i ? streams[j] : null;
        childSelections[j] = selectionChildIndices[j] == i ? selections[j] : null;
      }
      long selectPositionUs =
          periods[i].selectTracks(
              childSelections, mayRetainStreamFlags, childStreams, streamResetFlags, positionUs);
      if (i == 0) {
        positionUs = selectPositionUs;
      } else if (selectPositionUs != positionUs) {
        throw new IllegalStateException("Children enabled at different positions.");
      }
      boolean periodEnabled = false;
      for (int j = 0; j < selections.length; j++) {
        if (selectionChildIndices[j] == i) {
          // Assert that the child provided a stream for the selection.
          SampleStream childStream = Assertions.checkNotNull(childStreams[j]);
          newStreams[j] = childStreams[j];
          periodEnabled = true;
          streamPeriodIndices.put(childStream, i);
        } else if (streamChildIndices[j] == i) {
          // Assert that the child cleared any previous stream.
          Assertions.checkState(childStreams[j] == null);
        }
      }
      if (periodEnabled) {
        enabledPeriodsList.add(periods[i]);
      }
    }
    // Copy the new streams back into the streams array.
    System.arraycopy(newStreams, 0, streams, 0, newStreams.length);
    // Update the local state.
    enabledPeriods = enabledPeriodsList.toArray(new MediaPeriod[0]);
    compositeSequenceableLoader =
        compositeSequenceableLoaderFactory.createCompositeSequenceableLoader(enabledPeriods);
    return positionUs;
  }

  @Override
  public void discardBuffer(long positionUs, boolean toKeyframe) {
    for (MediaPeriod period : enabledPeriods) {
      period.discardBuffer(positionUs, toKeyframe);
    }
  }

  @Override
  public void reevaluateBuffer(long positionUs) {
    compositeSequenceableLoader.reevaluateBuffer(positionUs);
  }

  @Override
  public boolean continueLoading(long positionUs) {
    if (!childrenPendingPreparation.isEmpty()) {
      // Preparation is still going on.
      int childrenPendingPreparationSize = childrenPendingPreparation.size();
      for (int i = 0; i < childrenPendingPreparationSize; i++) {
        childrenPendingPreparation.get(i).continueLoading(positionUs);
      }
      return false;
    } else {
      return compositeSequenceableLoader.continueLoading(positionUs);
    }
  }

  @Override
  public boolean isLoading() {
    return compositeSequenceableLoader.isLoading();
  }

  @Override
  public long getNextLoadPositionUs() {
    return compositeSequenceableLoader.getNextLoadPositionUs();
  }

  @Override
  public long readDiscontinuity() {
    long discontinuityUs = C.TIME_UNSET;
    for (MediaPeriod period : enabledPeriods) {
      long otherDiscontinuityUs = period.readDiscontinuity();
      if (otherDiscontinuityUs != C.TIME_UNSET) {
        if (discontinuityUs == C.TIME_UNSET) {
          discontinuityUs = otherDiscontinuityUs;
          // First reported discontinuity. Seek all previous periods to the new position.
          for (MediaPeriod previousPeriod : enabledPeriods) {
            if (previousPeriod == period) {
              break;
            }
            if (previousPeriod.seekToUs(discontinuityUs) != discontinuityUs) {
              throw new IllegalStateException("Unexpected child seekToUs result.");
            }
          }
        } else if (otherDiscontinuityUs != discontinuityUs) {
          throw new IllegalStateException("Conflicting discontinuities.");
        }
      } else if (discontinuityUs != C.TIME_UNSET) {
        // We already have a discontinuity, seek this period to the new position.
        if (period.seekToUs(discontinuityUs) != discontinuityUs) {
          throw new IllegalStateException("Unexpected child seekToUs result.");
        }
      }
    }
    return discontinuityUs;
  }

  @Override
  public long getBufferedPositionUs() {
    return compositeSequenceableLoader.getBufferedPositionUs();
  }

  @Override
  public long seekToUs(long positionUs) {
    positionUs = enabledPeriods[0].seekToUs(positionUs);
    // Additional periods must seek to the same position.
    for (int i = 1; i < enabledPeriods.length; i++) {
      if (enabledPeriods[i].seekToUs(positionUs) != positionUs) {
        throw new IllegalStateException("Unexpected child seekToUs result.");
      }
    }
    return positionUs;
  }

  @Override
  public long getAdjustedSeekPositionUs(long positionUs, SeekParameters seekParameters) {
    MediaPeriod queryPeriod = enabledPeriods.length > 0 ? enabledPeriods[0] : periods[0];
    return queryPeriod.getAdjustedSeekPositionUs(positionUs, seekParameters);
  }

  // MediaPeriod.Callback implementation

  @Override
  public void onPrepared(MediaPeriod preparedPeriod) {
    childrenPendingPreparation.remove(preparedPeriod);
    if (!childrenPendingPreparation.isEmpty()) {
      return;
    }
    int totalTrackGroupCount = 0;
    for (MediaPeriod period : periods) {
      totalTrackGroupCount += period.getTrackGroups().length;
    }
    TrackGroup[] trackGroupArray = new TrackGroup[totalTrackGroupCount];
    int trackGroupIndex = 0;
    for (MediaPeriod period : periods) {
      TrackGroupArray periodTrackGroups = period.getTrackGroups();
      int periodTrackGroupCount = periodTrackGroups.length;
      for (int j = 0; j < periodTrackGroupCount; j++) {
        trackGroupArray[trackGroupIndex++] = periodTrackGroups.get(j);
      }
    }
    trackGroups = new TrackGroupArray(trackGroupArray);
    Assertions.checkNotNull(callback).onPrepared(this);
  }

  @Override
  public void onContinueLoadingRequested(MediaPeriod ignored) {
    Assertions.checkNotNull(callback).onContinueLoadingRequested(this);
  }

  private static final class TimeOffsetMediaPeriod implements MediaPeriod, MediaPeriod.Callback {

    private final MediaPeriod mediaPeriod;
    private final long timeOffsetUs;

    private @MonotonicNonNull Callback callback;

    public TimeOffsetMediaPeriod(MediaPeriod mediaPeriod, long timeOffsetUs) {
      this.mediaPeriod = mediaPeriod;
      this.timeOffsetUs = timeOffsetUs;
    }

    @Override
    public void prepare(Callback callback, long positionUs) {
      this.callback = callback;
      mediaPeriod.prepare(/* callback= */ this, positionUs - timeOffsetUs);
    }

    @Override
    public void maybeThrowPrepareError() throws IOException {
      mediaPeriod.maybeThrowPrepareError();
    }

    @Override
    public TrackGroupArray getTrackGroups() {
      return mediaPeriod.getTrackGroups();
    }

    @Override
    public List<StreamKey> getStreamKeys(List<ExoTrackSelection> trackSelections) {
      return mediaPeriod.getStreamKeys(trackSelections);
    }

    @Override
    public long selectTracks(
        @NullableType ExoTrackSelection[] selections,
        boolean[] mayRetainStreamFlags,
        @NullableType SampleStream[] streams,
        boolean[] streamResetFlags,
        long positionUs) {
      @NullableType SampleStream[] childStreams = new SampleStream[streams.length];
      for (int i = 0; i < streams.length; i++) {
        TimeOffsetSampleStream sampleStream = (TimeOffsetSampleStream) streams[i];
        childStreams[i] = sampleStream != null ? sampleStream.getChildStream() : null;
      }
      long startPositionUs =
          mediaPeriod.selectTracks(
              selections,
              mayRetainStreamFlags,
              childStreams,
              streamResetFlags,
              positionUs - timeOffsetUs);
      for (int i = 0; i < streams.length; i++) {
        @Nullable SampleStream childStream = childStreams[i];
        if (childStream == null) {
          streams[i] = null;
        } else if (streams[i] == null
            || ((TimeOffsetSampleStream) streams[i]).getChildStream() != childStream) {
          streams[i] = new TimeOffsetSampleStream(childStream, timeOffsetUs);
        }
      }
      return startPositionUs + timeOffsetUs;
    }

    @Override
    public void discardBuffer(long positionUs, boolean toKeyframe) {
      mediaPeriod.discardBuffer(positionUs - timeOffsetUs, toKeyframe);
    }

    @Override
    public long readDiscontinuity() {
      long discontinuityPositionUs = mediaPeriod.readDiscontinuity();
      return discontinuityPositionUs == C.TIME_UNSET
          ? C.TIME_UNSET
          : discontinuityPositionUs + timeOffsetUs;
    }

    @Override
    public long seekToUs(long positionUs) {
      return mediaPeriod.seekToUs(positionUs - timeOffsetUs) + timeOffsetUs;
    }

    @Override
    public long getAdjustedSeekPositionUs(long positionUs, SeekParameters seekParameters) {
      return mediaPeriod.getAdjustedSeekPositionUs(positionUs - timeOffsetUs, seekParameters)
          + timeOffsetUs;
    }

    @Override
    public long getBufferedPositionUs() {
      long bufferedPositionUs = mediaPeriod.getBufferedPositionUs();
      return bufferedPositionUs == C.TIME_END_OF_SOURCE
          ? C.TIME_END_OF_SOURCE
          : bufferedPositionUs + timeOffsetUs;
    }

    @Override
    public long getNextLoadPositionUs() {
      long nextLoadPositionUs = mediaPeriod.getNextLoadPositionUs();
      return nextLoadPositionUs == C.TIME_END_OF_SOURCE
          ? C.TIME_END_OF_SOURCE
          : nextLoadPositionUs + timeOffsetUs;
    }

    @Override
    public boolean continueLoading(long positionUs) {
      return mediaPeriod.continueLoading(positionUs - timeOffsetUs);
    }

    @Override
    public boolean isLoading() {
      return mediaPeriod.isLoading();
    }

    @Override
    public void reevaluateBuffer(long positionUs) {
      mediaPeriod.reevaluateBuffer(positionUs - timeOffsetUs);
    }

    @Override
    public void onPrepared(MediaPeriod mediaPeriod) {
      Assertions.checkNotNull(callback).onPrepared(/* mediaPeriod= */ this);
    }

    @Override
    public void onContinueLoadingRequested(MediaPeriod source) {
      Assertions.checkNotNull(callback).onContinueLoadingRequested(/* source= */ this);
    }
  }

  private static final class TimeOffsetSampleStream implements SampleStream {

    private final SampleStream sampleStream;
    private final long timeOffsetUs;

    public TimeOffsetSampleStream(SampleStream sampleStream, long timeOffsetUs) {
      this.sampleStream = sampleStream;
      this.timeOffsetUs = timeOffsetUs;
    }

    public SampleStream getChildStream() {
      return sampleStream;
    }

    @Override
    public boolean isReady() {
      return sampleStream.isReady();
    }

    @Override
    public void maybeThrowError() throws IOException {
      sampleStream.maybeThrowError();
    }

    @Override
    public int readData(
        FormatHolder formatHolder, DecoderInputBuffer buffer, @ReadFlags int readFlags) {
      int readResult = sampleStream.readData(formatHolder, buffer, readFlags);
      if (readResult == C.RESULT_BUFFER_READ) {
        buffer.timeUs = max(0, buffer.timeUs + timeOffsetUs);
      }
      return readResult;
    }

    @Override
    public int skipData(long positionUs) {
      return sampleStream.skipData(positionUs - timeOffsetUs);
    }
  }
}
