package com.ysz.dm.netty.dm.order.domain;

/**
 * @author carl
 */
public class MessageResp extends Message<OperationResult> {

  @Override
  public Class<? extends OperationResult> getMessageBodyDecodeClass(int opCode) {
    return OperationType.fromOpCode(opCode).getOperationResultClazz();
  }
}
