package com.ysz.dm.ddd.vshop.domain.domain.inventory.attrs.vip;

import com.ysz.dm.ddd.vshop.domain.domain.common.Rmb;
import com.ysz.dm.ddd.vshop.domain.domain.inventory.attrs.InventoryAttr;
import com.ysz.dm.ddd.vshop.domain.domain.inventory.attrs.InventoryAttrKey;
import java.util.Map;
import java.util.Optional;
import lombok.Getter;
import lombok.ToString;

/**
 * vip 场景下 的 attrs
 *
 * @author carl
 * @create 2022-10-25 3:51 PM
 **/
@ToString
@Getter
public final class VipInventoryAttrs {

  /**
   * months
   */
  private int months;
  /**
   * true 表示是 svip
   */
  private boolean svip;

  private Rmb vipPrice;
  private Rmb svipPrice;

  private VipInventoryAttrs() {
  }


  public static VipInventoryAttrs of(Map<InventoryAttrKey, InventoryAttr> origin) {
    VipInventoryAttrs vipInventoryAttrs = new VipInventoryAttrs();
    Optional<Integer> months = Optional.empty();
    Optional<Boolean> svip = Optional.empty();
    Optional<Rmb> vipPrice = Optional.empty();
    Optional<Rmb> svipPrice = Optional.empty();

    /*not great design*/
    return null;
  }

}
