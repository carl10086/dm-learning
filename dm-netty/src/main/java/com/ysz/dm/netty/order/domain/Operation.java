package com.ysz.dm.netty.order.domain;

/**
 * @author carl
 */
public abstract class Operation extends MessageBody {

  public abstract OperationResult execute();
}
