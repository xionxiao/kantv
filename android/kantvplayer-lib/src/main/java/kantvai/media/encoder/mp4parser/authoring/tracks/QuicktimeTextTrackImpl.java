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
package kantvai.media.encoder.mp4parser.authoring.tracks;

import kantvai.media.encoder.iso.boxes.*;
import kantvai.media.encoder.iso.boxes.sampleentry.TextSampleEntry;
import kantvai.media.encoder.mp4parser.authoring.AbstractTrack;
import kantvai.media.encoder.mp4parser.authoring.TrackMetaData;
import kantvai.media.encoder.mp4parser.boxes.apple.BaseMediaInfoAtom;
import kantvai.media.encoder.mp4parser.boxes.apple.GenericMediaHeaderAtom;
import kantvai.media.encoder.mp4parser.boxes.apple.GenericMediaHeaderTextAtom;
import kantvai.media.encoder.mp4parser.boxes.apple.QuicktimeTextSampleEntry;
import kantvai.media.encoder.mp4parser.boxes.threegpp26245.FontTableBox;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Collections;
import java.util.Date;
import java.util.LinkedList;
import java.util.List;

/**
 * A Text track as Quicktime Pro would create.
 */
public class QuicktimeTextTrackImpl extends AbstractTrack {
    TrackMetaData trackMetaData = new TrackMetaData();
    SampleDescriptionBox sampleDescriptionBox;
    List<Line> subs = new LinkedList<Line>();

    public List<Line> getSubs() {
        return subs;
    }

    public QuicktimeTextTrackImpl() {
        sampleDescriptionBox = new SampleDescriptionBox();
        QuicktimeTextSampleEntry textTrack = new QuicktimeTextSampleEntry();
        textTrack.setDataReferenceIndex(1);
        sampleDescriptionBox.addBox(textTrack);


        trackMetaData.setCreationTime(new Date());
        trackMetaData.setModificationTime(new Date());
        trackMetaData.setTimescale(1000);


    }


    public List<ByteBuffer> getSamples() {
        List<ByteBuffer> samples = new LinkedList<ByteBuffer>();
        long lastEnd = 0;
        for (Line sub : subs) {
            long silentTime = sub.from - lastEnd;
            if (silentTime > 0) {
                samples.add(ByteBuffer.wrap(new byte[]{0, 0}));
            } else if (silentTime < 0) {
                throw new Error("Subtitle display times may not intersect");
            }
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            DataOutputStream dos = new DataOutputStream(baos);
            try {
                dos.writeShort(sub.text.getBytes("UTF-8").length);
                dos.write(sub.text.getBytes("UTF-8"));
                dos.close();
            } catch (IOException e) {
                throw new Error("VM is broken. Does not support UTF-8");
            }
            samples.add(ByteBuffer.wrap(baos.toByteArray()));
            lastEnd = sub.to;
        }
        return samples;
    }

    public SampleDescriptionBox getSampleDescriptionBox() {
        return sampleDescriptionBox;
    }

    public List<TimeToSampleBox.Entry> getDecodingTimeEntries() {
        List<TimeToSampleBox.Entry> stts = new LinkedList<TimeToSampleBox.Entry>();
        long lastEnd = 0;
        for (Line sub : subs) {
            long silentTime = sub.from - lastEnd;
            if (silentTime > 0) {
                stts.add(new TimeToSampleBox.Entry(1, silentTime));
            } else if (silentTime < 0) {
                throw new Error("Subtitle display times may not intersect");
            }
            stts.add(new TimeToSampleBox.Entry(1, sub.to - sub.from));
            lastEnd = sub.to;
        }
        return stts;
    }

    public List<CompositionTimeToSample.Entry> getCompositionTimeEntries() {
        return null;
    }

    public long[] getSyncSamples() {
        return null;
    }

    public List<SampleDependencyTypeBox.Entry> getSampleDependencies() {
        return null;
    }

    public TrackMetaData getTrackMetaData() {
        return trackMetaData;
    }

    public String getHandler() {
        return "text";
    }


    public static class Line {
        long from;
        long to;
        String text;


        public Line(long from, long to, String text) {
            this.from = from;
            this.to = to;
            this.text = text;
        }

        public long getFrom() {
            return from;
        }

        public String getText() {
            return text;
        }

        public long getTo() {
            return to;
        }
    }

    public Box getMediaHeaderBox() {
        GenericMediaHeaderAtom ghmd = new GenericMediaHeaderAtom();
        ghmd.addBox(new BaseMediaInfoAtom());
        ghmd.addBox(new GenericMediaHeaderTextAtom());
        return ghmd;
    }

    public SubSampleInformationBox getSubsampleInformationBox() {
        return null;
    }
}
