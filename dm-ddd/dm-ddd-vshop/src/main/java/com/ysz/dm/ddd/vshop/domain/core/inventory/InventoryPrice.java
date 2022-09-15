package com.ysz.dm.ddd.vshop.domain.core.inventory;

import com.google.common.base.Preconditions;
import com.ysz.dm.ddd.vshop.domain.core.common.rmb.Rmb;
import lombok.Getter;
import lombok.ToString;

/**
 * @author carl
 * @create 2022-09-15 12:12 PM
 **/
@ToString
@Getter
public class InventoryPrice {

  private final Rmb settlePrice;

  private final Rmb salePrice;


  public InventoryPrice(Rmb settlePrice, Rmb salePrice) {
    this.settlePrice = Preconditions.checkNotNull(settlePrice);
    this.salePrice = Preconditions.checkNotNull(salePrice);
    Preconditions.checkArgument(this.settlePrice.compareTo(salePrice) >= 0, "settlePrice must >= salePrice");
  }
}
