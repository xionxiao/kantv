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
package kantvai.media.encoder.mp4parser.boxes.apple;

import kantvai.media.encoder.iso.boxes.Box;
import kantvai.media.encoder.iso.boxes.sampleentry.SampleEntry;

import java.nio.ByteBuffer;

public class TimeCodeBox extends SampleEntry {
    byte[] data;


    public TimeCodeBox() {
        super("tmcd");
    }

    @Override
    protected long getContentSize() {
        long size = 26;
        for (Box box : boxes) {
            size += box.getSize();
        }
        return size;
    }

    @Override
    public void _parseDetails(ByteBuffer content) {
        _parseReservedAndDataReferenceIndex(content);
        data = new byte[18];
        content.get(data);
        _parseChildBoxes(content);
    }

    @Override
    protected void getContent(ByteBuffer byteBuffer) {
        _writeReservedAndDataReferenceIndex(byteBuffer);
        byteBuffer.put(data);
        _writeChildBoxes(byteBuffer);
    }
}
