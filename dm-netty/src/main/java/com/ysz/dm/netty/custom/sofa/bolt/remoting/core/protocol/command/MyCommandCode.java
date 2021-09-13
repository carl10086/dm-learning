package com.ysz.dm.netty.custom.sofa.bolt.remoting.core.protocol.command;

import com.alipay.remoting.CommandCode;
import com.alipay.remoting.ProtocolCode;
import com.alipay.remoting.config.switches.ProtocolSwitch;
import com.alipay.remoting.exception.DeserializationException;
import com.alipay.remoting.exception.SerializationException;
import com.ysz.dm.netty.custom.sofa.bolt.remoting.core.protocol.MyInvokeContext;

public interface MyCommandCode {

  /**
   * 代表了唯一的 Protocol
   * @return
   */
  ProtocolCode getProtocolCode();

  /**
   *
   * 代表了唯一的 Command
   * @return command code
   */
  CommandCode getCmdCode();

  /**
   * Get the id of the command
   *
   * @return an int value represent the command id
   */
  int getId();

  /**
   * Get invoke context for this command
   *
   * @return context
   */
  MyInvokeContext getInvokeContext();

  /**
   * Get serializer type for this command
   *
   * @return
   */
  byte getSerializer();

  /**
   * Get the protocol switch status for this command
   *
   * @return
   */
  ProtocolSwitch getProtocolSwitch();

  /**
   * Serialize all parts of remoting command
   *
   * @throws SerializationException
   */
  void serialize() throws SerializationException;

  /**
   * Deserialize all parts of remoting command
   *
   * @throws DeserializationException
   */
  void deserialize() throws DeserializationException;

  /**
   * Serialize content of remoting command
   *
   * @param invokeContext
   * @throws SerializationException
   */
  void serializeContent(MyInvokeContext invokeContext) throws SerializationException;

  /**
   * Deserialize content of remoting command
   *
   * @param invokeContext
   * @throws DeserializationException
   */
  void deserializeContent(MyInvokeContext invokeContext) throws DeserializationException;
}
