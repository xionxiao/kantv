/*
 * Copyright (C) 2017 The Android Open Source Project
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
package kantvai.media.exoplayer2;

import kantvai.media.exoplayer2.Player.RepeatMode;

/** @deprecated Use a {@link ForwardingPlayer} or configure the player to customize operations. */
@Deprecated
public interface ControlDispatcher {

  /**
   * Dispatches a {@link Player#prepare()} operation.
   *
   * @param player The {@link Player} to which the operation should be dispatched.
   * @return True if the operation was dispatched. False if suppressed.
   */
  boolean dispatchPrepare(Player player);

  /**
   * Dispatches a {@link Player#setPlayWhenReady(boolean)} operation.
   *
   * @param player The {@link Player} to which the operation should be dispatched.
   * @param playWhenReady Whether playback should proceed when ready.
   * @return True if the operation was dispatched. False if suppressed.
   */
  boolean dispatchSetPlayWhenReady(Player player, boolean playWhenReady);

  /**
   * Dispatches a {@link Player#seekTo(int, long)} operation.
   *
   * @param player The {@link Player} to which the operation should be dispatched.
   * @param windowIndex The index of the window.
   * @param positionMs The seek position in the specified window, or {@link C#TIME_UNSET} to seek to
   *     the window's default position.
   * @return True if the operation was dispatched. False if suppressed.
   */
  boolean dispatchSeekTo(Player player, int windowIndex, long positionMs);

  /**
   * Dispatches a {@link Player#seekToPreviousWindow()} operation.
   *
   * @param player The {@link Player} to which the operation should be dispatched.
   * @return True if the operation was dispatched. False if suppressed.
   */
  boolean dispatchPrevious(Player player);

  /**
   * Dispatches a {@link Player#seekToNextWindow()} operation.
   *
   * @param player The {@link Player} to which the operation should be dispatched.
   * @return True if the operation was dispatched. False if suppressed.
   */
  boolean dispatchNext(Player player);

  /**
   * Dispatches a rewind operation.
   *
   * @param player The {@link Player} to which the operation should be dispatched.
   * @return True if the operation was dispatched. False if suppressed.
   */
  boolean dispatchRewind(Player player);

  /**
   * Dispatches a fast forward operation.
   *
   * @param player The {@link Player} to which the operation should be dispatched.
   * @return True if the operation was dispatched. False if suppressed.
   */
  boolean dispatchFastForward(Player player);

  /**
   * Dispatches a {@link Player#setRepeatMode(int)} operation.
   *
   * @param player The {@link Player} to which the operation should be dispatched.
   * @param repeatMode The repeat mode.
   * @return True if the operation was dispatched. False if suppressed.
   */
  boolean dispatchSetRepeatMode(Player player, @RepeatMode int repeatMode);

  /**
   * Dispatches a {@link Player#setShuffleModeEnabled(boolean)} operation.
   *
   * @param player The {@link Player} to which the operation should be dispatched.
   * @param shuffleModeEnabled Whether shuffling is enabled.
   * @return True if the operation was dispatched. False if suppressed.
   */
  boolean dispatchSetShuffleModeEnabled(Player player, boolean shuffleModeEnabled);

  /**
   * Dispatches a {@link Player#stop()} operation.
   *
   * @param player The {@link Player} to which the operation should be dispatched.
   * @param reset Whether the player should be reset.
   * @return True if the operation was dispatched. False if suppressed.
   */
  boolean dispatchStop(Player player, boolean reset);

  /**
   * Dispatches a {@link Player#setPlaybackParameters(PlaybackParameters)} operation.
   *
   * @param player The {@link Player} to which the operation should be dispatched.
   * @param playbackParameters The playback parameters.
   * @return True if the operation was dispatched. False if suppressed.
   */
  boolean dispatchSetPlaybackParameters(Player player, PlaybackParameters playbackParameters);

  /** Returns {@code true} if rewind is enabled, {@code false} otherwise. */
  boolean isRewindEnabled();

  /** Returns {@code true} if fast forward is enabled, {@code false} otherwise. */
  boolean isFastForwardEnabled();
}
