package com.ysz.dm.ddd.vshop.domain.core.inventory;

import com.ysz.dm.ddd.vshop.domain.core.common.sku.SkuId;
import lombok.Getter;
import lombok.ToString;

/**
 * @author carl
 * @create 2022-09-09 2:34 PM
 **/
@ToString
@Getter
public class InventorySku {

  /**
   * 对应的 sku
   */
  private SkuId id;

  private Integer quantity;


}
