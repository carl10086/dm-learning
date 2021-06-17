package com.ysz.dm.netty.custom.protocol.command;

import com.ysz.dm.netty.custom.protocol.constants.CustomCommandType;
import com.ysz.dm.netty.custom.protocol.constants.CustomSerializeType;
import com.ysz.dm.netty.custom.protocol.constants.CustomProtocolCodeType;
import java.io.Serializable;

/**
 * command 接口、包含了 request & response
 */
public interface CustomCommand extends Serializable {

  /**
   * 返回当前 protocol 的魔数
   * @return
   */
  CustomProtocolCodeType proto();

  /**
   *
   * @return 当前 command 的类型
   */
  CustomCommandType type();

  /**
   * 返回相对唯一的 id、暂时不知道啥用
   * @return
   */
  int id();


  CustomSerializeType serializeType();


  short classNameLength();


  int contentLength();

  byte[] className();


  byte[] content();
}
