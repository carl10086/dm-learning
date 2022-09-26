package com.ysz.dm.fast.basic.onjava.enums;

import com.ysz.dm.fast.basic.onjava.base.Enums;

/**
 * <pre>
 *  随机菜单. 核心是  枚举套了枚举.
 *
 *  被嵌套的枚举中 有相同的类型 .
 * </pre>
 *
 * @author carl.yu
 * @createAt 2022/9/26
 **/
public enum Meal {

  APPETIZER(Food.Appetizer.class),
  MAINCOURSE(Food.MainCourse.class),
  DESSERT(Food.Dessert.class),
  COFFEE(Food.Coffee.class);

  public interface Food {

    enum Appetizer implements Food {
      SALAD, SOUP, SPRING_ROLLS;
    }

    enum MainCourse implements Food {
      LASAGNE, BURRITO, PAD_THAI,
      LENTILS, HUMMUS, VINDALOO;
    }

    enum Dessert implements Food {
      TIRAMISU, GELATO, BLACK_FOREST_CAKE,
      FRUIT, CREME_CARAMEL;
    }

    enum Coffee implements Food {
      BLACK_COFFEE, DECAF_COFFEE, ESPRESSO,
      LATTE, CAPPUCCINO, TEA, HERB_TEA;
    }
  }

  private Food[] values;

  Meal(Class<? extends Food> kind) {
    this.values = kind.getEnumConstants();
  }

  public Food randomSelection() {
    return Enums.random(this.values);
  }


  public static void main(String[] args) {
    for (int i = 0; i < 5; i++) {
      for (Meal meal : Meal.values()) {
        Food food = meal.randomSelection();
        System.out.println(food);
      }
      System.out.println("***");
    }
  }
}
