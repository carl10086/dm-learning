package com.ysz.dm.ddd.vshop.domain.domain.inventory.handler;

import com.ysz.dm.ddd.vshop.domain.domain.inventory.attrs.builtin.InventoryMemoryAttrs;
import com.ysz.dm.ddd.vshop.domain.domain.inventory.cate.InventoryCateId;

/**
 * @author carl
 * @create 2022-10-27 10:36 AM
 **/
public interface InventoryHandler {

  /**
   * return the order of handler
   */
  int order();

  InventoryMemoryAttrs handle(InventoryCateId id);
}
