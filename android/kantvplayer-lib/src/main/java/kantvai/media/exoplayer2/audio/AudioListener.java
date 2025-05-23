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
package kantvai.media.exoplayer2.audio;

import kantvai.media.exoplayer2.Player;

/**
 * A listener for changes in audio configuration.
 *
 * @deprecated Use {@link Player.Listener}.
 */
@Deprecated
public interface AudioListener {

  /**
   * Called when the audio session ID changes.
   *
   * @param audioSessionId The audio session ID.
   */
  default void onAudioSessionIdChanged(int audioSessionId) {}

  /**
   * Called when the audio attributes change.
   *
   * @param audioAttributes The audio attributes.
   */
  default void onAudioAttributesChanged(AudioAttributes audioAttributes) {}

  /**
   * Called when the volume changes.
   *
   * @param volume The new volume, with 0 being silence and 1 being unity gain.
   */
  default void onVolumeChanged(float volume) {}

  /**
   * Called when skipping silences is enabled or disabled in the audio stream.
   *
   * @param skipSilenceEnabled Whether skipping silences in the audio stream is enabled.
   */
  default void onSkipSilenceEnabledChanged(boolean skipSilenceEnabled) {}
}
