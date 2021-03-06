package com.konakart.app;

import com.konakart.appif.*;

/**
 *  The KonaKart Custom Engine - RemoveStockReservationsForOrderProducts - Generated by CreateKKCustomEng
 */
@SuppressWarnings("all")
public class RemoveStockReservationsForOrderProducts
{
    KKEng kkEng = null;

    /**
     * Constructor
     */
     public RemoveStockReservationsForOrderProducts(KKEng _kkEng)
     {
         kkEng = _kkEng;
     }

     public int removeStockReservationsForOrderProducts(String sessionId, OrderProductIf[] orderProducts, StockReservationOptionsIf options) throws KKException
     {
         return kkEng.removeStockReservationsForOrderProducts(sessionId, orderProducts, options);
     }
}
