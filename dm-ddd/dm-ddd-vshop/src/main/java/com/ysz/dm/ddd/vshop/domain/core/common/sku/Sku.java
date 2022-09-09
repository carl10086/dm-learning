package com.ysz.dm.ddd.vshop.domain.core.common.sku;

import com.ysz.dm.ddd.vshop.domain.core.common.media.PictureId;
import java.util.List;
import lombok.Getter;
import lombok.ToString;

/**
 * @author carl
 * @create 2022-09-09 2:32 PM
 **/
@ToString
@Getter
public class Sku {

  private List<PictureId> pics;


  private SkuStatus status;

  /**
   * sku id
   */
  private SkuId id;

  /**
   * 单位
   */
  private SkuUnit unit;



}
