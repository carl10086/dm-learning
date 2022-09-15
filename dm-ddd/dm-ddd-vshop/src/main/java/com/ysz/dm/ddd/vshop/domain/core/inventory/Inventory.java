package com.ysz.dm.ddd.vshop.domain.core.inventory;

import com.ysz.dm.ddd.vshop.domain.core.common.lang.CreateUpdate;
import com.ysz.dm.ddd.vshop.domain.core.common.richtext.RichText;
import com.ysz.dm.ddd.vshop.domain.core.inventory.cate.InventoryCateId;
import com.ysz.dm.ddd.vshop.domain.core.inventory.cate.InventoryCateProperties;
import com.ysz.dm.ddd.vshop.domain.core.inventory.tag.InventoryTagId;
import java.util.List;
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
   * 类目 id, 决定了 商品的动态属性
   */
  private InventoryCateId cateId;

  /**
   * 商品详情, 提前生成的富文本, 建议 用 cdn 优化
   */
  private RichText inventoryDetail;

  /**
   * 规格参数, 提前生成的富文本, 建议 用 cdn 优化
   */
  private RichText inventoryParams;

  /**
   * 商品价格
   */
  private InventoryPrice price;

  /**
   * 商品 sku, 可覆盖 部分参数 .
   */
  private List<InventorySku> skus;

  private List<InventoryTagId> tags;


  private InventoryCateProperties inventoryCateProperties;

  public Inventory setId(InventoryId id) {
    this.id = id;
    return this;
  }

  public Inventory setTitle(InventoryTitle title) {
    this.title = title;
    return this;
  }

  public Inventory setCateId(InventoryCateId cateId) {
    this.cateId = cateId;
    return this;
  }

  public Inventory setInventoryDetail(RichText inventoryDetail) {
    this.inventoryDetail = inventoryDetail;
    return this;
  }

  public Inventory setInventoryParams(RichText inventoryParams) {
    this.inventoryParams = inventoryParams;
    return this;
  }

  public Inventory setPrice(InventoryPrice price) {
    this.price = price;
    return this;
  }

  public Inventory setSkus(List<InventorySku> skus) {
    this.skus = skus;
    return this;
  }

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

  public Inventory setInventoryCateProperties(InventoryCateProperties inventoryCateProperties) {
    this.inventoryCateProperties = inventoryCateProperties;
    return this;
  }
}
