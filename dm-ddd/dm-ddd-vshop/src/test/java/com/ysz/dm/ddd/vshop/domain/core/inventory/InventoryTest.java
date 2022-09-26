package com.ysz.dm.ddd.vshop.domain.core.inventory;

import com.google.common.collect.Lists;
import com.ysz.dm.ddd.vshop.domain.core.common.lang.MapBuilder;
import com.ysz.dm.ddd.vshop.domain.core.common.richtext.RichText;
import com.ysz.dm.ddd.vshop.domain.core.common.richtext.RichTextType;
import com.ysz.dm.ddd.vshop.domain.core.common.rmb.Rmb;
import com.ysz.dm.ddd.vshop.domain.core.inventory.cate.InventoryCate;
import com.ysz.dm.ddd.vshop.domain.core.inventory.cate.InventoryCateId;
import com.ysz.dm.ddd.vshop.domain.core.inventory.cate.InventoryCateProperties;
import com.ysz.dm.ddd.vshop.domain.core.inventory.cate.InventoryCatePropertyKey;
import com.ysz.dm.ddd.vshop.domain.core.inventory.cate.InventoryCatePropertyKeyId;
import com.ysz.dm.ddd.vshop.domain.core.inventory.cate.InventoryCatePropertyType;
import java.util.ArrayList;
import java.util.List;
import java.util.SortedSet;
import java.util.TreeSet;
import org.junit.Test;

/**
 * @author carl
 * @create 2022-09-15 11:48 AM
 **/
public class InventoryTest {

  @Test
  public void vip() throws Exception {
    /*1. 创建 vip 分类*/
    InventoryCateId cateId = new InventoryCateId(2L);

    InventoryCate cate = new InventoryCate(cateId);
    SortedSet<InventoryCatePropertyKey> keys = new TreeSet<>();
    cate.setKeys(keys);

    keys.add(new InventoryCatePropertyKey()
                 .setName("是否是 svip")
                 .setType(InventoryCatePropertyType.selected)
                 .setValueClass(String.class)
                 .setSelectedValues(Lists.newArrayList("是", "否"))
                 .setId(new InventoryCatePropertyKeyId(1L)));
    keys.add(new InventoryCatePropertyKey()
                 .setName("时间区间")
                 .setType(InventoryCatePropertyType.selected)
                 .setValueClass(String.class)
                 .setSelectedValues(Lists.newArrayList("1个月", "3个月", "12个月"))
                 .setId(new InventoryCatePropertyKeyId(2L)));
    keys.add(new InventoryCatePropertyKey()
                 .setName("是否订阅")
                 .setType(InventoryCatePropertyType.selected)
                 .setValueClass(String.class)
                 .setSelectedValues(Lists.newArrayList("是", "否"))
                 .setId(new InventoryCatePropertyKeyId(3L)));


    /*2. 增加商品 vip*/
    Inventory vip = new Inventory();
    vip
        .setId(new InventoryId(1L))
        .setPrice(new InventoryPrice(Rmb.ofYuan(10L), Rmb.ofYuan(15L)))
        .setTitle(new InventoryTitle("1个月 VIP"))
        .setInventoryCateProperties(new InventoryCateProperties(
            new MapBuilder<InventoryCatePropertyKeyId, Object>()
                .put(new InventoryCatePropertyKeyId(1L), "是")
                .put(new InventoryCatePropertyKeyId(2L), "1个月")
                .put(new InventoryCatePropertyKeyId(3L), "否")
                .build()
        ));

    Inventory svip = new Inventory();
    svip
        .setId(new InventoryId(1L))
        .setPrice(new InventoryPrice(Rmb.ofYuan(10L), Rmb.ofYuan(15L)))
        .setTitle(new InventoryTitle("1个月 SVIP"))
        .setInventoryCateProperties(new InventoryCateProperties(
            new MapBuilder<InventoryCatePropertyKeyId, Object>()
                .put(new InventoryCatePropertyKeyId(1L), "是")
                .put(new InventoryCatePropertyKeyId(2L), "1个月")
                .put(new InventoryCatePropertyKeyId(3L), "是")
                .build()
        ));


  }


  @Test
  public void iphone14() throws Exception {
    Inventory inventory = new Inventory();

    /*1. 创建 keys & cate */
    SortedSet<InventoryCatePropertyKey> keys = new TreeSet<>();
    keys.add(
        new InventoryCatePropertyKey<String>()
            .setId(new InventoryCatePropertyKeyId(1L))
            .setName("套餐类型")
            .setDesc("")
            .setType(InventoryCatePropertyType.selected)
            .setValueClass(String.class)
            .setSelectedValues(
                Lists.newArrayList("官方标配", "套餐二")
            ));

    keys.add(
        new InventoryCatePropertyKey<String>()
            .setId(new InventoryCatePropertyKeyId(2L))
            .setName("存储容量")
            .setDesc("")
            .setType(InventoryCatePropertyType.selected)
            .setValueClass(String.class)
            .setSelectedValues(
                Lists.newArrayList("128GB", "256GB")
            ));

    keys.add(
        new InventoryCatePropertyKey<String>()
            .setId(new InventoryCatePropertyKeyId(3L))
            .setName("机身颜色")
            .setDesc("")
            .setType(InventoryCatePropertyType.selected)
            .setValueClass(String.class)
            .setSelectedValues(
                Lists.newArrayList("午夜色", "星光色", "蓝色", "紫色", "红色")
            ));

    InventoryCateId cateId = new InventoryCateId(1L);
    InventoryCate cate = new InventoryCate(cateId);
    cate.setKeys(keys);
    inventory.setCateId(cateId);


    /*2. 添加 sku*/
    List<InventorySku> skus = new ArrayList<>();
    inventory.setSkus(skus);
    skus.add(
        new InventorySku(new SkuId(1L))
            .setPrice(new InventoryPrice(Rmb.ofYuan(6000), Rmb.ofYuan(5999)))
            .setInventoryCateProperties(
                new InventoryCateProperties(
                    new MapBuilder<InventoryCatePropertyKeyId, Object>()
                        .put(new InventoryCatePropertyKeyId(1L), "官方标配")
                        .put(new InventoryCatePropertyKeyId(2L), "128GB")
                        .put(new InventoryCatePropertyKeyId(3L), "午夜色")
                        .build())
            )
    );

    skus.add(
        new InventorySku(new SkuId(1L))
            .setPrice(new InventoryPrice(Rmb.ofYuan(8000), Rmb.ofYuan(7899)))
            .setInventoryCateProperties(
                new InventoryCateProperties(
                    new MapBuilder<InventoryCatePropertyKeyId, Object>()
                        .put(new InventoryCatePropertyKeyId(1L), "官方标配")
                        .put(new InventoryCatePropertyKeyId(2L), "256GB")
                        .put(new InventoryCatePropertyKeyId(3L), "午夜色")
                        .build())
            )
    );

    /*3. inventorDetail*/
    inventory.setInventoryDetail(new RichText()
                                     .setType(RichTextType.markdown)
                                     .setContent(new StringBuilder()
                                                     .append("- 品牌名称: Apple/苹果")
                                                     .append("- 证书编号: 2021011606410117")
                                                     .append("- 证书状态: 有效")
                                                     .toString()));


    /*4. inventory Params*/
    inventory.setInventoryParams(new RichText()
                                     .setType(RichTextType.html)
                                     .setContent(new StringBuilder()
                                                     .append("<ul>")
                                                     .append("<li><b>拍照功能:</b> 后置摄像头 1200万</li>")
                                                     .append("</ul>")
                                                     .toString())
    );
  }

}