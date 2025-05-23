/*
 * Copyright 2012 Sebastian Annies, Hamburg
 *
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package kantvai.media.encoder.iso;

import kantvai.media.encoder.iso.boxes.Box;
import kantvai.media.encoder.iso.boxes.ContainerBox;

import java.io.IOException;
import java.nio.channels.ReadableByteChannel;

/**
 * Basic interface to create boxes from a <code>IsoBufferWrapper</code> and its parent.
 */
public interface BoxParser {
    Class<? extends Box> getClassForFourCc(String type, byte[] userType, String parent);

    Box parseBox(ReadableByteChannel in, ContainerBox parent) throws IOException;
}
