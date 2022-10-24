package com.ysz.dm.ddd.vshop.domain.domain.inventory.inventory;

import com.ysz.dm.ddd.vshop.domain.domain.inventory.cate.InventoryCateTest;
import org.junit.Before;
import org.junit.Test;

/**
 * @author carl
 * @create 2022-10-24 6:18 PM
 **/
public class InventoryTest {

  private InventoryCateTest.Root cate;

  @Before
  public void setUp() throws Exception {
    this.cate = InventoryCateTest.mockRoot();
  }


  @Test
  public void testCreateVip() {
    var id = 1L;
    Inventory vip = new Inventory();
    vip.setId(new InventoryId(id));
    vip.setCateId(this.cate.vip().getId());
  }
}