package com.ysz.dm.ddd.vshop.domain.core.inventory;

import com.ysz.dm.ddd.vshop.domain.core.common.CreateUpdate;
import com.ysz.dm.ddd.vshop.domain.core.common.richtext.RichText;
import lombok.Getter;
import lombok.ToString;

/**
 * @author carl
 * @create 2022-09-09 10:23 AM
 **/
@ToString
@Getter
public class Inventory {

  /**
   * 商品 id
   */
  private InventoryId id;

  /**
   * 标题
   */
  private InventoryTitle title;


  /**
   * 每个商品有自己的协议， 这里可能是用 markdown 或者 html 展示的 协议
   */
  private RichText agreement;

  /**
   * 商品价格
   */
  private InventoryPrices prices;


  /**
   * 商品预售时间
   */
  private InventorySaleTime saleTime;



  /**
   * 商品相关图片
   */
  private InventoryPictures pictures;

  private CreateUpdate createUpdate;

}
