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
package kantvai.media.exoplayer2.source.dash;

import android.os.SystemClock;
import androidx.annotation.Nullable;
import kantvai.media.exoplayer2.Format;
import kantvai.media.exoplayer2.source.chunk.ChunkSource;
import kantvai.media.exoplayer2.source.dash.PlayerEmsgHandler.PlayerTrackEmsgHandler;
import kantvai.media.exoplayer2.source.dash.manifest.DashManifest;
import kantvai.media.exoplayer2.trackselection.ExoTrackSelection;
import kantvai.media.exoplayer2.upstream.LoaderErrorThrower;
import kantvai.media.exoplayer2.upstream.TransferListener;
import java.util.List;

/** A {@link ChunkSource} for DASH streams. */
public interface DashChunkSource extends ChunkSource {

  /** Factory for {@link DashChunkSource}s. */
  interface Factory {

    /**
     * @param manifestLoaderErrorThrower Throws errors affecting loading of manifests.
     * @param manifest The initial manifest.
     * @param baseUrlExclusionList The base URL exclusion list.
     * @param periodIndex The index of the corresponding period in the manifest.
     * @param adaptationSetIndices The indices of the corresponding adaptation sets in the period.
     * @param trackSelection The track selection.
     * @param elapsedRealtimeOffsetMs If known, an estimate of the instantaneous difference between
     *     server-side unix time and {@link SystemClock#elapsedRealtime()} in milliseconds,
     *     specified as the server's unix time minus the local elapsed time. Or {@link
     *     kantvai.media.exoplayer2.C#TIME_UNSET} if unknown.
     * @param enableEventMessageTrack Whether to output an event message track.
     * @param closedCaptionFormats The {@link Format Formats} of closed caption tracks to be output.
     * @param transferListener The transfer listener which should be informed of any data transfers.
     *     May be null if no listener is available.
     * @return The created {@link DashChunkSource}.
     */
    DashChunkSource createDashChunkSource(
        LoaderErrorThrower manifestLoaderErrorThrower,
        DashManifest manifest,
        BaseUrlExclusionList baseUrlExclusionList,
        int periodIndex,
        int[] adaptationSetIndices,
        ExoTrackSelection trackSelection,
        int type,
        long elapsedRealtimeOffsetMs,
        boolean enableEventMessageTrack,
        List<Format> closedCaptionFormats,
        @Nullable PlayerTrackEmsgHandler playerEmsgHandler,
        @Nullable TransferListener transferListener);
  }

  /**
   * Updates the manifest.
   *
   * @param newManifest The new manifest.
   */
  void updateManifest(DashManifest newManifest, int periodIndex);

  /**
   * Updates the track selection.
   *
   * @param trackSelection The new track selection instance. Must be equivalent to the previous one.
   */
  void updateTrackSelection(ExoTrackSelection trackSelection);
}
