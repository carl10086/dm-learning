package com.ysz.dm.ddd.vshop.domain.domain.inventory.attrs;

import com.ysz.dm.ddd.vshop.domain.domain.inventory.attrs.builtin.InventoryVipAttrs;
import com.ysz.dm.ddd.vshop.domain.domain.inventory.attrs.builtin.InventoryVirtualAttrs;
import java.util.Optional;

/**
 * @author carl
 * @create 2022-10-25 4:20 PM
 **/
public interface InventoryAttrs {

  /**
   * @return vip attrs
   */
  Optional<InventoryVipAttrs> vip();

  /**
   * @return virtual attrs
   */
  Optional<InventoryVirtualAttrs> virtual();

}
