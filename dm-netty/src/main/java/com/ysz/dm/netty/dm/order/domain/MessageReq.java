package com.ysz.dm.netty.dm.order.domain;

/**
 * @author carl
 */
public class MessageReq extends Message<Operation> {

  @Override
  public Class<? extends Operation> getMessageBodyDecodeClass(int opCode) {
    return OperationType.fromOpCode(opCode)
        .getOperationClazz();
  }


  public static MessageReq of(long id, Operation operation) {
    MessageReq messageReq = new MessageReq();
    messageReq.setMessageHeader(new MessageHeader(
        1,
        OperationType.fromOperation(operation).getOpCode(),
        id
    ));
    messageReq.setMessageBody(operation);
    return messageReq;

  }
}
