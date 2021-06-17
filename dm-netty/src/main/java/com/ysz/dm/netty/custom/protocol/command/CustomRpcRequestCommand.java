package com.ysz.dm.netty.custom.protocol.command;

import com.ysz.dm.netty.common.JsonTools;
import com.ysz.dm.netty.custom.protocol.CustomCommandIdGenerator;
import com.ysz.dm.netty.custom.protocol.constants.CustomCommandType;
import com.ysz.dm.netty.custom.protocol.constants.CustomSerializeType;
import com.ysz.dm.netty.custom.protocol.constants.CustomProtocolCodeType;
import java.nio.charset.StandardCharsets;

public class CustomRpcRequestCommand implements CustomCommand {

  private Integer requestId = CustomCommandIdGenerator.nextId();
  /**
   * rpc 请求 server 参数
   */
  private Object requestObject;


  private transient byte[] requestObjectAsBytes;
  /**
   * rpc 请求 server 端参数类名
   */
  private String requestClass;

  private transient byte[] requestClassAsBytes;


  @Override
  public CustomProtocolCodeType proto() {
    return CustomProtocolCodeType.V1;
  }

  @Override
  public CustomCommandType type() {
    return CustomCommandType.RPC_REQUEST;
  }

  @Override
  public int id() {
    return requestId;
  }

  @Override
  public CustomSerializeType serializeType() {
    return CustomSerializeType.json;
  }

  @Override
  public short classNameLength() {
    return (short) className().length;
  }

  @Override
  public int contentLength() {
    return content().length;
  }

  @Override
  public byte[] className() {
    if (requestClassAsBytes == null) {
      requestClassAsBytes = requestClass.getBytes(StandardCharsets.UTF_8);
    }
    return requestClassAsBytes;
  }


  @Override
  public byte[] content() {
    if (requestObjectAsBytes == null) {
      requestObjectAsBytes = JsonTools.toJson(requestObject).getBytes(StandardCharsets.UTF_8);
    }

    return requestClassAsBytes;
  }
}
