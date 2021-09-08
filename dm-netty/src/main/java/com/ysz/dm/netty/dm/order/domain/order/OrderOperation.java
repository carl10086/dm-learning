package com.ysz.dm.netty.dm.order.domain.order;

import com.ysz.dm.netty.dm.order.domain.Operation;
import com.ysz.dm.netty.dm.order.domain.OperationResult;
import lombok.Data;

/**
 * @author carl
 */
@Data
public class OrderOperation extends Operation {

  private final Integer tableId;
  private final String dish;


  @Override
  public OperationResult execute() {
    System.out.println("order's executing startup with orderReq:" + toString());
    System.out.println("order's executing complete");
    return new OrderOperationResult(
        tableId,
        dish,
        true
    );
  }
}
