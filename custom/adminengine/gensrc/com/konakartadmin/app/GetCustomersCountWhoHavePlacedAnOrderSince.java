package com.konakartadmin.app;

import java.util.Calendar;
import com.konakartadmin.bl.KKAdmin;

/**
 *  The KonaKart Custom Engine - GetCustomersCountWhoHavePlacedAnOrderSince - Generated by CreateKKAdminCustomEng
 */
@SuppressWarnings("all")
public class GetCustomersCountWhoHavePlacedAnOrderSince
{
    KKAdmin kkAdminEng = null;

    /**
     * Constructor
     */
     public GetCustomersCountWhoHavePlacedAnOrderSince(KKAdmin _kkAdminEng)
     {
         kkAdminEng = _kkAdminEng;
     }

     public int getCustomersCountWhoHavePlacedAnOrderSince(String sessionId, Calendar lastOrderDate) throws KKAdminException
     {
         return kkAdminEng.getCustomersCountWhoHavePlacedAnOrderSince(sessionId, lastOrderDate);
     }
}