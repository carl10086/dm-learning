package com.ysz.dm.ddd.vshop.domain.core.inventory;

import java.util.List;

/**
 * @author carl
 * @create 2022-09-14 12:26 PM
 **/
public class InventoryCatePropertyKey {

  /**
   * true 表示是可选择的项目
   */
  private boolean optional;

  private InventoryCatePropertyType type;

  private List<Object> selectedValues;

}
