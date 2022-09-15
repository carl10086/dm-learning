package com.ysz.dm.ddd.vshop.domain.core.inventory;

import java.time.Instant;
import lombok.Getter;
import lombok.ToString;

/**
 * <pre>
 *  商品销售时间
 * </pre>
 *
 * @author carl.yu
 * @createAt 2022/9/9
 **/
@ToString
@Getter
public class InventorySaleTime {


  /**
   * 开始销售时间
   */
  private final Instant saleStartAt;

  /**
   * 销售结束时间
   */
  private final Instant saleEndAt;

  public InventorySaleTime(Instant saleStartAt, Instant saleEndAt) {
    this.saleStartAt = saleStartAt;
    this.saleEndAt = saleEndAt;
  }
}
