/*
 * Copyright 2020 The Android Open Source Project
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
package kantvai.media.exoplayer2.source.dash.manifest;

import kantvai.media.exoplayer2.C;

/** Represents a service description element. */
public final class ServiceDescriptionElement {

  /** The target live offset in milliseconds, or {@link C#TIME_UNSET} if undefined. */
  public final long targetOffsetMs;
  /** The minimum live offset in milliseconds, or {@link C#TIME_UNSET} if undefined. */
  public final long minOffsetMs;
  /** The maximum live offset in milliseconds, or {@link C#TIME_UNSET} if undefined. */
  public final long maxOffsetMs;
  /**
   * The minimum factor by which playback can be sped up for live speed adjustment, or {@link
   * C#RATE_UNSET} if undefined.
   */
  public final float minPlaybackSpeed;
  /**
   * The maximum factor by which playback can be sped up for live speed adjustment, or {@link
   * C#RATE_UNSET} if undefined.
   */
  public final float maxPlaybackSpeed;

  /**
   * Creates a service description element.
   *
   * @param targetOffsetMs The target live offset in milliseconds, or {@link C#TIME_UNSET} if
   *     undefined.
   * @param minOffsetMs The minimum live offset in milliseconds, or {@link C#TIME_UNSET} if
   *     undefined.
   * @param maxOffsetMs The maximum live offset in milliseconds, or {@link C#TIME_UNSET} if
   *     undefined.
   * @param minPlaybackSpeed The minimum factor by which playback can be sped up for live speed
   *     adjustment, or {@link C#RATE_UNSET} if undefined.
   * @param maxPlaybackSpeed The maximum factor by which playback can be sped up for live speed
   *     adjustment, or {@link C#RATE_UNSET} if undefined.
   */
  public ServiceDescriptionElement(
      long targetOffsetMs,
      long minOffsetMs,
      long maxOffsetMs,
      float minPlaybackSpeed,
      float maxPlaybackSpeed) {
    this.targetOffsetMs = targetOffsetMs;
    this.minOffsetMs = minOffsetMs;
    this.maxOffsetMs = maxOffsetMs;
    this.minPlaybackSpeed = minPlaybackSpeed;
    this.maxPlaybackSpeed = maxPlaybackSpeed;
  }
}
