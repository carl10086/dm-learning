package com.ysz.dm.ddd.vshop.domain.domain.inventory.inventory;

import java.time.Instant;
import java.util.Optional;

/**
 * @author carl
 * @create 2022-10-24 6:28 PM
 **/
public record InventorySaleTime(
    Optional<Instant> startAt,
    Optional<Instant> endAt
) {

}
