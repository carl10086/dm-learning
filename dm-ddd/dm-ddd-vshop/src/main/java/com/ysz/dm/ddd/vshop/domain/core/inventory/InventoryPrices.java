package com.ysz.dm.ddd.vshop.domain.core.inventory;

import com.ysz.dm.ddd.vshop.domain.core.common.price.Rmb;
import lombok.Getter;
import lombok.ToString;

/**
 * <pre>
 * 商品是有价格策略的
 * </pre>
 *
 * @author carl.yu
 * @createAt 2022/9/9
 **/
@ToString
@Getter
public class InventoryPrices {

  private final Rmb settlePrice;

  private final Rmb salePrice;

  public InventoryPrices(Rmb settlePrice, Rmb salePrice) {
    this.settlePrice = settlePrice;
    this.salePrice = salePrice;
  }
}
