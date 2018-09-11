package com.konakartadmin.app;

import com.konakartadmin.appif.*;
import com.konakartadmin.bl.KKAdmin;

/**
 *  The KonaKart Custom Engine - ImportDigitalDownload - Generated by CreateKKAdminCustomEng
 */
@SuppressWarnings("all")
public class ImportDigitalDownload
{
    KKAdmin kkAdminEng = null;

    /**
     * Constructor
     */
     public ImportDigitalDownload(KKAdmin _kkAdminEng)
     {
         kkAdminEng = _kkAdminEng;
     }

     public int importDigitalDownload(String sessionId, AdminDigitalDownload digDownload) throws KKAdminException
     {
         return kkAdminEng.importDigitalDownload(sessionId, digDownload);
     }
}
