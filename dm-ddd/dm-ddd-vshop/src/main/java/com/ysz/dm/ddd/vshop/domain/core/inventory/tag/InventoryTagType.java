package com.ysz.dm.ddd.vshop.domain.core.inventory.tag;

import lombok.Getter;
import lombok.ToString;

/**
 * @author carl
 * @create 2022-09-15 5:12 PM
 **/
@Getter
@ToString
public enum InventoryTagType {
  /**
   * 单件商品, 限时优惠
   */
  single_limit_in_time(false),

  /**
   * 单件商品, 首单优惠
   */
  single_first_order(false),

  /**
   * 多件商品 满减
   */
  multi_full_cut(true);


  /**
   * 商品表 是否支持多个商品
   */
  private final boolean multi;

  InventoryTagType(boolean multi) {
    this.multi = multi;
  }
}
