package kantvai.media.encoder.packets;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import kantvai.media.encoder.amf.AmfString;
import kantvai.media.encoder.io.ChunkStreamInfo;

/**
 * AMF Data packet
 * 
 * Also known as NOTIFY in some RTMP implementations.
 * 
 * The client or the server sends this message to send Metadata or any user data
 * to the peer. Metadata includes details about the data (audio, video etc.) 
 * like creation time, duration, theme and so on.
 * 
 * @author francois
 */
public class Data extends VariableBodyRtmpPacket {

    private String type;

    public Data(RtmpHeader header) {
        super(header);
    }

    public Data(String type) {
        super(new RtmpHeader(RtmpHeader.ChunkType.TYPE_0_FULL, ChunkStreamInfo.RTMP_CID_OVER_CONNECTION, RtmpHeader.MessageType.DATA_AMF0));
        this.type = type;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    @Override
    public void readBody(InputStream in) throws IOException {
        // Read notification type
        type = AmfString.readStringFrom(in, false);
        int bytesRead = AmfString.sizeOf(type, false);
        // Read data body
        readVariableData(in, bytesRead);
    }

    /** 
     * This method is public for Data to make it easy to dump its contents to 
     * another output stream
     */
    @Override
    protected void writeBody(OutputStream out) throws IOException {
        AmfString.writeStringTo(out, type, false);
        writeVariableData(out);
    }

    @Override
    protected byte[] array() {
        return null;
    }

    @Override
    protected int size() {
        return 0;
    }
}
