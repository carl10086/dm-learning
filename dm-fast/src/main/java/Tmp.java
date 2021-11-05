public class Tmp {

  public static void main(String[] args) throws Exception {
    final CrossThreadObj crossThreadObj = new CrossThreadObj();
  }


  /**
   * 跨线程对象 , 假设有 100 个属性
   */
  private static class CrossThreadObj {

  }
}
