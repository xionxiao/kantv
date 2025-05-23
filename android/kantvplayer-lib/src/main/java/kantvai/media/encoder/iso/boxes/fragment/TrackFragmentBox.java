/*
 * Copyright 2009 castLabs GmbH, Berlin
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

package kantvai.media.encoder.iso.boxes.fragment;

import kantvai.media.encoder.mp4parser.AbstractContainerBox;
import kantvai.media.encoder.iso.boxes.Box;

/**
 * aligned(8) class TrackFragmentBox extends Box('traf'){
 * }
 */
public class TrackFragmentBox extends AbstractContainerBox {
    public static final String TYPE = "traf";

    public TrackFragmentBox() {
        super(TYPE);
    }


    public TrackFragmentHeaderBox getTrackFragmentHeaderBox() {
        for (Box box : getBoxes()) {
            if (box instanceof TrackFragmentHeaderBox) {
                return (TrackFragmentHeaderBox) box;
            }
        }
        return null;
    }

}
