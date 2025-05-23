package kantvai.media.encoder.mp4parser.boxes.piff;

import kantvai.media.encoder.iso.IsoFile;
import kantvai.media.encoder.iso.IsoTypeReader;
import kantvai.media.encoder.iso.IsoTypeWriter;
import kantvai.media.encoder.mp4parser.AbstractFullBox;
import kantvai.media.encoder.mp4parser.util.Path;
import kantvai.media.encoder.mp4parser.util.UUIDConverter;

import java.io.FileInputStream;
import java.io.IOException;
import java.lang.Override;import java.lang.String;import java.lang.StringBuilder;import java.nio.ByteBuffer;
import java.util.UUID;

import static kantvai.media.encoder.mp4parser.util.CastUtils.l2i;

/**
 * aligned(8) class UuidBasedProtectionSystemSpecificHeaderBox extends FullBox(‘uuid’,
 * extended_type=0xd08a4f18-10f3-4a82-b6c8-32d8aba183d3,
 * version=0, flags=0)
 * {
 * unsigned int(8)[16] SystemID;
 * unsigned int(32) DataSize;
 * unsigned int(8)[DataSize] Data;
 * }
 */
public class UuidBasedProtectionSystemSpecificHeaderBox extends AbstractFullBox {
    public static byte[] USER_TYPE = new byte[]{(byte) 0xd0, (byte) 0x8a, 0x4f, 0x18, 0x10, (byte) 0xf3, 0x4a, (byte) 0x82,
                (byte) 0xb6, (byte) 0xc8, 0x32, (byte) 0xd8, (byte) 0xab, (byte) 0xa1, (byte) 0x83, (byte) 0xd3};

    UUID systemId;

    ProtectionSpecificHeader protectionSpecificHeader;

    public UuidBasedProtectionSystemSpecificHeaderBox() {
        super("uuid", USER_TYPE);
    }

    @Override
    protected long getContentSize() {
        return 24 + protectionSpecificHeader.getData().limit();
    }

    @Override
    public byte[] getUserType() {
        return USER_TYPE;
    }

    @Override
    protected void getContent(ByteBuffer byteBuffer) {
        writeVersionAndFlags(byteBuffer);
        IsoTypeWriter.writeUInt64(byteBuffer, systemId.getMostSignificantBits());
        IsoTypeWriter.writeUInt64(byteBuffer, systemId.getLeastSignificantBits());
        ByteBuffer data = protectionSpecificHeader.getData();
        data.rewind();
        IsoTypeWriter.writeUInt32(byteBuffer, data.limit());
        byteBuffer.put(data);
    }

    @Override
    protected void _parseDetails(ByteBuffer content) {
        parseVersionAndFlags(content);
        byte[] systemIdBytes = new byte[16];
        content.get(systemIdBytes);
        systemId = UUIDConverter.convert(systemIdBytes);
        int dataSize = l2i(IsoTypeReader.readUInt32(content));
        protectionSpecificHeader = ProtectionSpecificHeader.createFor(systemId, content);
    }

    public UUID getSystemId() {
        return systemId;
    }

    public void setSystemId(UUID systemId) {
        this.systemId = systemId;
    }

    public String getSystemIdString() {
        return systemId.toString();
    }

    public ProtectionSpecificHeader getProtectionSpecificHeader() {
        return protectionSpecificHeader;
    }

    public String getProtectionSpecificHeaderString() {
        return protectionSpecificHeader.toString();
    }

    public void setProtectionSpecificHeader(ProtectionSpecificHeader protectionSpecificHeader) {
        this.protectionSpecificHeader = protectionSpecificHeader;
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder();
        sb.append("UuidBasedProtectionSystemSpecificHeaderBox");
        sb.append("{systemId=").append(systemId.toString());
        sb.append(", dataSize=").append(protectionSpecificHeader.getData().limit());
        sb.append('}');
        return sb.toString();
    }



}
