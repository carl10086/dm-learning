package com.ysz.dm.ddd.vshop.domain.domain.inventory.cate;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import lombok.Getter;
import lombok.ToString;

/**
 * @author carl
 * @create 2022-10-27 10:32 AM
 **/
@ToString
@Getter
public class InventoryCateProps {

  private Map<InventoryCatePropId, InventoryCateProp> props;
  private Map<String, InventoryCateProp> nameToPropMap;

  public InventoryCateProps(Collection<InventoryCateProp> props) {
    this.props = new HashMap<>(props.size());
    this.nameToPropMap = new HashMap<>(props.size());

    for (InventoryCateProp prop : props) {
      this.props.put(prop.id(), prop);
      this.nameToPropMap.put(prop.name(), prop);
    }
  }
}
