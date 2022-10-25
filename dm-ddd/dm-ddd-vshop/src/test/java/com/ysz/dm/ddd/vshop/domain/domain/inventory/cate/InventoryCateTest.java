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
    var propId = 10L;

    var root = new InventoryCate(InventoryCateIds.root(new InventoryCateId(cateId)));
    var virtual = new InventoryCate(InventoryCateIds.levelTwo(root.id(), new InventoryCateId(cateId++)))
        .props(new InventoryCateProps(Lists.newArrayList(
            new InventoryCateProp(propId++, "虚拟商品类型").optional(false)
                .selectedValues(Lists.newArrayList("vip", "素材", "图片")),
            new InventoryCateProp(propId++, "vip价格").optional(true).sku(false).type(InventoryCatePropType.plaintext),
            new InventoryCateProp(propId++, "svip价格").optional(true).sku(false).type(InventoryCatePropType.plaintext)
        )));

    var vip = new InventoryCate(InventoryCateIds.levelThree(root.id(), virtual.id(), new InventoryCateId(cateId++)))
        .props(new InventoryCateProps(Lists.newArrayList(
            new InventoryCateProp(propId++, "是否是svip").selectedValues(Lists.newArrayList("是", "否")),
            new InventoryCateProp(propId++, "持续时间").selectedValues(Lists.newArrayList("1个月", "3个月"))
        )));

    return new Root(
        root,
        virtual,
        vip
    );
  }

  @Test
  public void rootCate() {
    Root root = mockRoot();
    log.info("root.vip:{}", root);
  }

  public record Root(
      InventoryCate root,
      InventoryCate virtual,
      InventoryCate vip
  ) {

    @Override
    public String toString() {
      return "Root{\n" +
          "root=" + root +
          "\nvirtual=" + virtual +
          "\n vip=" + vip +
          '}';
    }
  }

}