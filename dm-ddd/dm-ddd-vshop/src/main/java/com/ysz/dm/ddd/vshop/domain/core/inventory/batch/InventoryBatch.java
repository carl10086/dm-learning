package com.ysz.dm.ddd.vshop.domain.core.inventory.batch;

import lombok.Getter;
import lombok.ToString;

/**
 * 场, 活动批次 id
 *
 * @author carl
 * @create 2022-09-15 5:14 PM
 **/
@ToString
@Getter
public class InventoryBatch {

  private final String batchNo;

  public InventoryBatch(String batchNo) {
    this.batchNo = batchNo;
  }
}
