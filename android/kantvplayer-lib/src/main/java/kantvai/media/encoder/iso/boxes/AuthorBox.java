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
import kantvai.media.encoder.iso.Utf8;
import kantvai.media.encoder.mp4parser.AbstractFullBox;

import java.nio.ByteBuffer;

/**
 * Meta information in a 'udta' box about a track.
 * Defined in 3GPP 26.244.
 *
 * @see kantvai.media.encoder.iso.boxes.UserDataBox
 */
public class AuthorBox extends AbstractFullBox {
    public static final String TYPE = "auth";

    private String language;
    private String author;

    public AuthorBox() {
        super(TYPE);
    }

    /**
     * Declares the language code for the {@link #getAuthor()} return value. See ISO 639-2/T for the set of three
     * character codes.Each character is packed as the difference between its ASCII value and 0x60. The code is
     * confined to being three lower-case letters, so these values are strictly positive.
     *
     * @return the language code
     */
    public String getLanguage() {
        return language;
    }

    /**
     * Author information.
     *
     * @return the author
     */
    public String getAuthor() {
        return author;
    }

    public void setLanguage(String language) {
        this.language = language;
    }

    public void setAuthor(String author) {
        this.author = author;
    }

    protected long getContentSize() {
        return 7 + Utf8.utf8StringLengthInBytes(author);
    }

    @Override
    public void _parseDetails(ByteBuffer content) {
        parseVersionAndFlags(content);
        language = IsoTypeReader.readIso639(content);
        author = IsoTypeReader.readString(content);
    }

    @Override
    protected void getContent(ByteBuffer byteBuffer) {
        writeVersionAndFlags(byteBuffer);
        IsoTypeWriter.writeIso639(byteBuffer, language);
        byteBuffer.put(Utf8.convert(author));
        byteBuffer.put((byte) 0);
    }


    public String toString() {
        return "AuthorBox[language=" + getLanguage() + ";author=" + getAuthor() + "]";
    }
}
