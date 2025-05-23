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

package kantvai.media.encoder.iso.boxes.vodafone;

import kantvai.media.encoder.iso.IsoTypeReader;
import kantvai.media.encoder.iso.Utf8;
import kantvai.media.encoder.mp4parser.AbstractFullBox;

import java.nio.ByteBuffer;

/**
 * A vodafone specific box.
 */
public class CoverUriBox extends AbstractFullBox {
    public static final String TYPE = "cvru";

    private String coverUri;

    public CoverUriBox() {
        super(TYPE);
    }

    public String getCoverUri() {
        return coverUri;
    }

    public void setCoverUri(String coverUri) {
        this.coverUri = coverUri;
    }

    protected long getContentSize() {
        return Utf8.utf8StringLengthInBytes(coverUri) + 5;
    }

    @Override
    public void _parseDetails(ByteBuffer content) {
        parseVersionAndFlags(content);
        coverUri = IsoTypeReader.readString(content);
    }

    @Override
    protected void getContent(ByteBuffer byteBuffer) {
        writeVersionAndFlags(byteBuffer);
        byteBuffer.put(Utf8.convert(coverUri));
        byteBuffer.put((byte) 0);
    }


    public String toString() {
        return "CoverUriBox[coverUri=" + getCoverUri() + "]";
    }
}
