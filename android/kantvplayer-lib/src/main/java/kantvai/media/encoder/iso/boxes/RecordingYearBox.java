/*  
 * Copyright 2008 CoreMedia AG, Hamburg
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

package kantvai.media.encoder.iso.boxes;

import kantvai.media.encoder.iso.IsoTypeReader;
import kantvai.media.encoder.iso.IsoTypeWriter;
import kantvai.media.encoder.mp4parser.AbstractFullBox;

import java.nio.ByteBuffer;

/**
 *
 */
public class RecordingYearBox extends AbstractFullBox {
    public static final String TYPE = "yrrc";

    int recordingYear;

    public RecordingYearBox() {
        super(TYPE);
    }


    protected long getContentSize() {
        return 6;
    }

    public int getRecordingYear() {
        return recordingYear;
    }

    public void setRecordingYear(int recordingYear) {
        this.recordingYear = recordingYear;
    }


    @Override
    public void _parseDetails(ByteBuffer content) {
        parseVersionAndFlags(content);
        recordingYear = IsoTypeReader.readUInt16(content);
    }

    @Override
    protected void getContent(ByteBuffer byteBuffer) {
        writeVersionAndFlags(byteBuffer);
        IsoTypeWriter.writeUInt16(byteBuffer, recordingYear);
    }

}
