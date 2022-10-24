package com.ysz.dm.ddd.vshop.domain.domain.inventory.cate;

import com.google.common.collect.Lists;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

/**
 * @author carl
 * @create 2022-10-24 6:02 PM
 **/
@Slf4j
public class InventoryCateTest {

  public static Root mockRoot() {
    var cateId = 1L;
    var propId = 1L;

    var root = new InventoryCate(new InventoryCateId(cateId++), null);
    var virtual = new InventoryCate(new InventoryCateId(cateId++), root.getId());
    virtual.setProps(Lists.newArrayList(
        new InventoryCateProp("是否virtual商品", new InventoryCatePropId(propId++))
            .setSelectedValues(Lists.newArrayList("是", "否")),
        new InventoryCateProp("虚拟商品类型", new InventoryCatePropId(propId++))
            .setSelectedValues(Lists.newArrayList("vip", "素材", "图片"))
    ));

    var vip = new InventoryCate(new InventoryCateId(cateId++), virtual.getId())
        .setProps(Lists.newArrayList(
            new InventoryCateProp("vip 时长", new InventoryCatePropId(propId++))
                .setSelectedValues(Lists.newArrayList("1个月", "2个月", "3个月"))
        ));

    return new Root(
        root,
        virtual,
        vip
    );
  }

  @Test
  public void rootCate() {
    Root root = mockRoot();
    log.info("root.vip:{}", root.vip());
  }

  public record Root(
      InventoryCate root,
      InventoryCate virtual,
      InventoryCate vip
  ) {

  }

}