package com.ysz.dm.ddd.vshop.domain.application.command.order;

import com.ysz.dm.ddd.vshop.domain.domain.inventory.inventory.InventoryId;
import com.ysz.dm.ddd.vshop.domain.domain.order.OrderType;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.ToString;

/**
 * @author carl
 * @create 2022-10-25 2:32 PM
 **/
@Getter
@ToString
@AllArgsConstructor
@NoArgsConstructor
public class CreateOrderCommand {

  private long userId;
  private long createAt;
  private OrderType type;
  /**
   * only support one inventory
   */
  private InventoryId inventoryId;

  public CreateOrderCommand setUserId(long userId) {
    this.userId = userId;
    return this;
  }

  public CreateOrderCommand setCreateAt(long createAt) {
    this.createAt = createAt;
    return this;
  }

  public CreateOrderCommand setType(OrderType type) {
    this.type = type;
    return this;
  }

  public CreateOrderCommand setInventoryId(InventoryId inventoryId) {
    this.inventoryId = inventoryId;
    return this;
  }
}
