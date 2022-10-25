package com.ysz.dm.ddd.vshop.domain.application.command.order;

import com.ysz.dm.ddd.vshop.domain.domain.inventory.inventory.InventoryId;
import com.ysz.dm.ddd.vshop.domain.domain.order.OrderType;
import lombok.Builder;
import lombok.Getter;
import lombok.NonNull;
import lombok.ToString;

/**
 * @author carl
 * @create 2022-10-25 2:32 PM
 **/
@Getter
@ToString
@Builder
public class CreateOrderCommand {

  private @NonNull Long userId;
  private @NonNull Long createAt;
  private @NonNull OrderType type;
  /**
   * only support one inventory
   */
  private @NonNull InventoryId inventoryId;

}
