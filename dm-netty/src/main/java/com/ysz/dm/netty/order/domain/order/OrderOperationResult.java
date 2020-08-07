package com.ysz.dm.netty.order.domain.order;

import com.ysz.dm.netty.order.domain.OperationResult;
import lombok.Data;

/**
 * @author carl
 */
@Data
public class OrderOperationResult extends OperationResult {

  private final Integer tableId;

  private final String dish;

  private final Boolean complete;

}
