package com.ysz.dm.ddd.vshop.domain.core.inventory.tag;

import com.ysz.dm.ddd.vshop.domain.core.common.lang.ActiveStatus;
import com.ysz.dm.ddd.vshop.domain.core.common.lang.CreateUpdate;
import com.ysz.dm.ddd.vshop.domain.core.common.lang.JsonString;
import com.ysz.dm.ddd.vshop.domain.core.common.richtext.RichText;
import com.ysz.dm.ddd.vshop.domain.core.inventory.batch.InventoryBatch;
import lombok.Getter;
import lombok.ToString;
import org.apache.commons.lang3.Range;

/**
 * 商品标 系统
 *
 * @author carl
 * @create 2022-09-15 5:10 PM
 **/
@ToString
@Getter
public class InventoryTag<V> {

  /**
   * 商品标唯一 id
   */
  private InventoryTagId id;

  /**
   * 商品场次 id
   */
  private InventoryBatch batch;


  /**
   * 商品标名称
   */
  private RichText tagName;

  /**
   * active status
   */
  private ActiveStatus activeStatus;

  /**
   * active time
   */
  private Range<Long> active;

  /**
   * 修改时间
   */
  private CreateUpdate createUpdate;

  /**
   * 商品标类型
   */
  private InventoryTagType tagType;

  /**
   * 商品标动态细节
   */
  private JsonString<V> tagDetail;


  public InventoryTag<V> setId(InventoryTagId id) {
    this.id = id;
    return this;
  }

  public InventoryTag<V> setBatch(InventoryBatch batch) {
    this.batch = batch;
    return this;
  }

  public InventoryTag<V> setTagName(RichText tagName) {
    this.tagName = tagName;
    return this;
  }

  public InventoryTag<V> setActiveStatus(ActiveStatus activeStatus) {
    this.activeStatus = activeStatus;
    return this;
  }

  public InventoryTag<V> setActive(Range<Long> active) {
    this.active = active;
    return this;
  }

  public InventoryTag<V> setCreateUpdate(CreateUpdate createUpdate) {
    this.createUpdate = createUpdate;
    return this;
  }

  public InventoryTag<V> setTagType(InventoryTagType tagType) {
    this.tagType = tagType;
    return this;
  }

  public InventoryTag<V> setTagDetail(JsonString<V> tagDetail) {
    this.tagDetail = tagDetail;
    return this;
  }
}
