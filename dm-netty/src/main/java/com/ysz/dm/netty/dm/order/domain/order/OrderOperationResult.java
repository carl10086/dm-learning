package com.ysz.dm.netty.dm.order.domain.order;

import com.ysz.dm.netty.dm.order.domain.OperationResult;
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
