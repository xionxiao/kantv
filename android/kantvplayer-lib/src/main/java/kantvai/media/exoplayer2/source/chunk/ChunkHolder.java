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
package kantvai.media.exoplayer2.source.chunk;

import androidx.annotation.Nullable;

/** Holds a chunk or an indication that the end of the stream has been reached. */
public final class ChunkHolder {

  /** The chunk. */
  @Nullable public Chunk chunk;

  /** Indicates that the end of the stream has been reached. */
  public boolean endOfStream;

  /** Clears the holder. */
  public void clear() {
    chunk = null;
    endOfStream = false;
  }
}
