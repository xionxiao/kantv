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
package kantvai.media.exoplayer2.device;

import kantvai.media.exoplayer2.Player;

/**
 * A listener for changes of {@link DeviceInfo} or device volume.
 *
 * @deprecated Use {@link Player.Listener}.
 */
@Deprecated
public interface DeviceListener {

  /** Called when the device information changes. */
  default void onDeviceInfoChanged(DeviceInfo deviceInfo) {}

  /** Called when the device volume or mute state changes. */
  default void onDeviceVolumeChanged(int volume, boolean muted) {}
}
