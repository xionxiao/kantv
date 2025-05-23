/*
 * Copyright (C) 2018 The Android Open Source Project
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
package kantvai.media.exoplayer2.trackselection;

import android.os.SystemClock;
import androidx.annotation.Nullable;
import kantvai.media.exoplayer2.source.TrackGroupArray;
import kantvai.media.exoplayer2.trackselection.DefaultTrackSelector.SelectionOverride;
import kantvai.media.exoplayer2.trackselection.ExoTrackSelection.Definition;
import kantvai.media.exoplayer2.upstream.LoadErrorHandlingPolicy;
import kantvai.media.exoplayer2.util.MimeTypes;
import org.checkerframework.checker.nullness.compatqual.NullableType;

/** Track selection related utility methods. */
public final class TrackSelectionUtil {

  private TrackSelectionUtil() {}

  /** Functional interface to create a single adaptive track selection. */
  public interface AdaptiveTrackSelectionFactory {

    /**
     * Creates an adaptive track selection for the provided track selection definition.
     *
     * @param trackSelectionDefinition A {@link Definition} for the track selection.
     * @return The created track selection.
     */
    ExoTrackSelection createAdaptiveTrackSelection(Definition trackSelectionDefinition);
  }

  /**
   * Creates track selections for an array of track selection definitions, with at most one
   * multi-track adaptive selection.
   *
   * @param definitions The list of track selection {@link Definition definitions}. May include null
   *     values.
   * @param adaptiveTrackSelectionFactory A factory for the multi-track adaptive track selection.
   * @return The array of created track selection. For null entries in {@code definitions} returns
   *     null values.
   */
  public static @NullableType ExoTrackSelection[] createTrackSelectionsForDefinitions(
      @NullableType Definition[] definitions,
      AdaptiveTrackSelectionFactory adaptiveTrackSelectionFactory) {
    ExoTrackSelection[] selections = new ExoTrackSelection[definitions.length];
    boolean createdAdaptiveTrackSelection = false;
    for (int i = 0; i < definitions.length; i++) {
      Definition definition = definitions[i];
      if (definition == null) {
        continue;
      }
      if (definition.tracks.length > 1 && !createdAdaptiveTrackSelection) {
        createdAdaptiveTrackSelection = true;
        selections[i] = adaptiveTrackSelectionFactory.createAdaptiveTrackSelection(definition);
      } else {
        selections[i] =
            new FixedTrackSelection(
                definition.group, definition.tracks[0], /* type= */ definition.type);
      }
    }
    return selections;
  }

  /**
   * Updates {@link DefaultTrackSelector.Parameters} with an override.
   *
   * @param parameters The current {@link DefaultTrackSelector.Parameters} to build upon.
   * @param rendererIndex The renderer index to update.
   * @param trackGroupArray The {@link TrackGroupArray} of the renderer.
   * @param isDisabled Whether the renderer should be set disabled.
   * @param override An optional override for the renderer. If null, no override will be set and an
   *     existing override for this renderer will be cleared.
   * @return The updated {@link DefaultTrackSelector.Parameters}.
   */
  public static DefaultTrackSelector.Parameters updateParametersWithOverride(
      DefaultTrackSelector.Parameters parameters,
      int rendererIndex,
      TrackGroupArray trackGroupArray,
      boolean isDisabled,
      @Nullable SelectionOverride override) {
    DefaultTrackSelector.ParametersBuilder builder =
        parameters
            .buildUpon()
            .clearSelectionOverrides(rendererIndex)
            .setRendererDisabled(rendererIndex, isDisabled);
    if (override != null) {
      builder.setSelectionOverride(rendererIndex, trackGroupArray, override);
    }
    return builder.build();
  }

  /** Returns if a {@link TrackSelectionArray} has at least one track of the given type. */
  public static boolean hasTrackOfType(TrackSelectionArray trackSelections, int trackType) {
    for (int i = 0; i < trackSelections.length; i++) {
      @Nullable TrackSelection trackSelection = trackSelections.get(i);
      if (trackSelection == null) {
        continue;
      }
      for (int j = 0; j < trackSelection.length(); j++) {
        if (MimeTypes.getTrackType(trackSelection.getFormat(j).sampleMimeType) == trackType) {
          return true;
        }
      }
    }
    return false;
  }

  /**
   * Returns the {@link LoadErrorHandlingPolicy.FallbackOptions} with the tracks of the given {@link
   * ExoTrackSelection} and with a single location option indicating that there are no alternative
   * locations available.
   *
   * @param trackSelection The track selection to get the number of total and excluded tracks.
   * @return The {@link LoadErrorHandlingPolicy.FallbackOptions} for the given track selection.
   */
  public static LoadErrorHandlingPolicy.FallbackOptions createFallbackOptions(
      ExoTrackSelection trackSelection) {
    long nowMs = SystemClock.elapsedRealtime();
    int numberOfTracks = trackSelection.length();
    int numberOfExcludedTracks = 0;
    for (int i = 0; i < numberOfTracks; i++) {
      if (trackSelection.isBlacklisted(i, nowMs)) {
        numberOfExcludedTracks++;
      }
    }
    return new LoadErrorHandlingPolicy.FallbackOptions(
        /* numberOfLocations= */ 1,
        /* numberOfExcludedLocations= */ 0,
        numberOfTracks,
        numberOfExcludedTracks);
  }
}
