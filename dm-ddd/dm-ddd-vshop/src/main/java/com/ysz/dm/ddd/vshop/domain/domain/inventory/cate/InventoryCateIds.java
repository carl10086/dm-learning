package com.ysz.dm.ddd.vshop.domain.domain.inventory.cate;

import java.util.Optional;
import lombok.NonNull;

/**
 * @author carl
 * @create 2022-10-27 11:07 AM
 **/
public record InventoryCateIds(
    /*层级*/
    InventoryCateLevel level, InventoryCateId one, Optional<InventoryCateId> two, Optional<InventoryCateId> three) {

  public InventoryCateId currentId() {
    switch (level) {
      case one -> {
        return this.one();
      }
      case two -> {
        return this.two().get();
      }
      case three -> {
        return this.three().get();
      }
      default -> {
        throw new IllegalStateException();
      }
    }
  }


  public static InventoryCateIds root(InventoryCateId id) {
    return new InventoryCateIds(InventoryCateLevel.one, id, Optional.empty(), Optional.empty());
  }

  public static InventoryCateIds levelTwo(
      @NonNull InventoryCateId one, @NonNull InventoryCateId two
  ) {
    return new InventoryCateIds(InventoryCateLevel.two, one, Optional.of(two), Optional.empty());
  }


  public static InventoryCateIds levelThree(
      @NonNull InventoryCateId one, @NonNull InventoryCateId two, @NonNull InventoryCateId three
  ) {
    return new InventoryCateIds(InventoryCateLevel.two, one, Optional.of(two), Optional.of(three));
  }

}
